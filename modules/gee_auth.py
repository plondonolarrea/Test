"""
modules/gee_auth.py
Autenticación con Google Earth Engine (GEE) usando la API de Python.

Modos soportados
────────────────
  "saved"           → usa credenciales OAuth ya guardadas en disco
                      (~/.config/earthengine/credentials)
                      Requiere haber ejecutado previamente en la terminal:
                          earthengine authenticate
  "interactive"     → lanza flujo OAuth en el navegador del servidor
                      (útil solo en entornos locales)
  "service_account" → usa clave JSON de un Service Account de GCP

setup_gee() devuelve (ok: bool, msg: str) para mostrar feedback
exacto en la UI sin ocultar el error real.
"""

import logging
import os

import ee

from config import GEE_PROJECT, SERVICE_ACCOUNT_EMAIL, SERVICE_ACCOUNT_KEY_FILE

logger = logging.getLogger(__name__)

# GEE solo puede inicializarse una vez por proceso Python.
# Este flag evita errores de "already initialized".
_gee_initialized = False


def _reset():
    """Fuerza reset del estado interno de la SDK de GEE (solo para testing)."""
    global _gee_initialized
    _gee_initialized = False


# ── Modo 1: credenciales guardadas (flujo normal de producción) ────────────────
def initialize_from_saved_credentials(project: str = GEE_PROJECT) -> tuple[bool, str]:
    """
    Inicializa GEE con el token OAuth ya almacenado en disco.
    Si las credenciales no existen el error lo indicará claramente.

    Retorna (True, "ok") o (False, "mensaje de error").
    """
    global _gee_initialized
    try:
        if _gee_initialized:
            # Ya inicializado en esta sesión → verificar conectividad
            ee.Number(1).getInfo()
            return True, "GEE ya estaba inicializado en esta sesión."
        ee.Initialize(project=project)
        # Prueba de conectividad rápida
        ee.Number(1).getInfo()
        _gee_initialized = True
        logger.info("GEE inicializado desde credenciales guardadas. Proyecto: %s", project)
        return True, f"GEE conectado correctamente. Proyecto: {project}"
    except FileNotFoundError:
        msg = (
            "No se encontraron credenciales guardadas. "
            "Ejecuta este comando en la terminal y vuelve a intentarlo:\n"
            "    earthengine authenticate"
        )
        logger.error(msg)
        return False, msg
    except Exception as exc:
        msg = f"Error al inicializar GEE: {exc}"
        logger.error(msg)
        return False, msg


# ── Modo 2: OAuth interactivo ─────────────────────────────────────────────────
def authenticate_interactive(project: str = GEE_PROJECT) -> tuple[bool, str]:
    """
    Lanza el flujo OAuth. En entorno local abre el navegador; en servidores
    imprime un URL de autorización en la consola.

    NOTA: Preferible hacerlo desde la terminal antes de iniciar la app:
          earthengine authenticate
    """
    global _gee_initialized
    try:
        ee.Authenticate(force=True)
        ee.Initialize(project=project)
        ee.Number(1).getInfo()
        _gee_initialized = True
        logger.info("GEE autenticado (OAuth interactivo). Proyecto: %s", project)
        return True, f"GEE autenticado correctamente. Proyecto: {project}"
    except Exception as exc:
        msg = f"Error en autenticación OAuth: {exc}"
        logger.error(msg)
        return False, msg


# ── Modo 3: Service Account ────────────────────────────────────────────────────
def authenticate_service_account(
    key_file: str = SERVICE_ACCOUNT_KEY_FILE,
    email: str = SERVICE_ACCOUNT_EMAIL,
    project: str = GEE_PROJECT,
) -> tuple[bool, str]:
    """Inicializa GEE con un Service Account de GCP."""
    global _gee_initialized
    if not key_file:
        return False, "Debes especificar la ruta al archivo JSON del Service Account."
    if not os.path.isfile(key_file):
        return False, f"Archivo de Service Account no encontrado: {key_file}"
    try:
        credentials = ee.ServiceAccountCredentials(email=email, key_file=key_file)
        ee.Initialize(credentials=credentials, project=project)
        ee.Number(1).getInfo()
        _gee_initialized = True
        logger.info("GEE autenticado con Service Account: %s", email)
        return True, f"GEE autenticado con Service Account. Proyecto: {project}"
    except Exception as exc:
        msg = f"Error con Service Account: {exc}"
        logger.error(msg)
        return False, msg


# ── Punto de entrada único ────────────────────────────────────────────────────
def setup_gee(
    mode: str = "saved",
    project: str = GEE_PROJECT,
    sa_email: str = "",
    sa_key_file: str = "",
) -> tuple[bool, str]:
    """
    Autentifica / inicializa GEE según el modo elegido.

    Retorna
    -------
    (bool, str) → (éxito, mensaje descriptivo para mostrar en la UI)
    """
    if not project or project.startswith("your-"):
        return False, (
            "El Project ID de GCP no está configurado. "
            "Edita config.py → GEE_PROJECT o escríbelo en el campo de la UI."
        )

    if mode == "service_account":
        return authenticate_service_account(sa_key_file, sa_email, project)
    if mode == "interactive":
        return authenticate_interactive(project)
    # "saved" (defecto) — NO hace fallback automático a interactivo
    return initialize_from_saved_credentials(project)


# ── Helpers ───────────────────────────────────────────────────────────────────
def is_initialized() -> bool:
    """Verifica si GEE responde correctamente en esta sesión."""
    try:
        ee.Number(1).getInfo()
        return True
    except Exception:
        return False


def get_gee_status() -> dict:
    """Estado de la conexión para mostrar en la UI."""
    if is_initialized():
        return {"connected": True,  "message": "GEE conectado correctamente."}
    return {"connected": False, "message": "GEE no inicializado."}
