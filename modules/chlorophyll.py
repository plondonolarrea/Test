"""
modules/chlorophyll.py
Cálculo de índices espectrales de clorofila y calidad del agua
a partir de imágenes Sentinel-2 SR.

Índices implementados
─────────────────────
NDCI    Normalized Difference Chlorophyll Index
        (Mishra & Mishra, 2012) → válido para aguas continentales
        = (B5 − B4) / (B5 + B4)

CHL_RE  Red-Edge Chlorophyll ratio
        = B5 / B4

NDWI    Normalized Difference Water Index (Gao, 1996)
        = (B3 − B8) / (B3 + B8)   [detección de agua]

FAI     Floating Algae Index (Hu, 2009)
        = B8 − [B4 + (B11 − B4) × (865 − 665) / (1610 − 665)]

B3_B2   Green/Blue ratio  (proxy de turbidez)
        = B3 / B2

NDTI    Normalized Difference Turbidity Index
        = (B4 − B3) / (B4 + B3)

Nota: todos los índices se calculan sobre compuestos ya escalados [0,1].
"""

import logging
from typing import Optional

import ee

from config import BANDS

logger = logging.getLogger(__name__)


# ── Versión SERVER-SIDE (sin getInfo(), segura dentro de .map()) ──────────────
def compute_index_server(image: ee.Image, index_name: str) -> ee.Image:
    """
    Calcula el índice directamente sobre los nombres de banda nativos de
    Sentinel-2 SR (sin getInfo()). Es la versión que se usa dentro de
    ee.ImageCollection.map() para extracción masiva en una sola llamada.
    """
    b = {k: image.select(v) for k, v in BANDS.items() if k != "qa"}

    if index_name == "NDCI":
        return b["red_edge1"].subtract(b["red"]) \
                             .divide(b["red_edge1"].add(b["red"])) \
                             .rename("NDCI")
    if index_name == "CHL_RE":
        return b["red_edge1"].divide(b["red"]).rename("CHL_RE")
    if index_name == "NDWI":
        return b["green"].subtract(b["nir"]) \
                         .divide(b["green"].add(b["nir"])) \
                         .rename("NDWI")
    if index_name == "FAI":
        slope    = (865 - 665) / (1610 - 665)
        baseline = b["red"].add(b["swir1"].subtract(b["red"]).multiply(slope))
        return b["nir"].subtract(baseline).rename("FAI")
    if index_name == "B3_B2":
        return b["green"].divide(b["blue"]).rename("B3_B2")
    if index_name == "NDTI":
        return b["red"].subtract(b["green"]) \
                       .divide(b["red"].add(b["green"])) \
                       .rename("NDTI")
    raise ValueError(f"Índice desconocido: {index_name}")


# ── Sufijo de bandas en compuestos (ee.ImageCollection.reduce()) ──────────────
# Cuando se aplica .median() sobre la colección, GEE agrega "_median" al nombre.
# Se usan helpers para resolver el nombre real de la banda.

def _band(image: ee.Image, key: str) -> ee.Image:
    """Selecciona una banda por su nombre base o con sufijo _median/_mean."""
    band_name = BANDS[key]
    all_bands = image.bandNames().getInfo()
    if band_name in all_bands:
        return image.select(band_name)
    # Buscar con sufijo
    for b in all_bands:
        if b.startswith(band_name):
            return image.select(b)
    raise ValueError(f"Banda '{band_name}' (key='{key}') no encontrada. Bandas disponibles: {all_bands}")


# ── Índices individuales ──────────────────────────────────────────────────────
def compute_ndci(image: ee.Image) -> ee.Image:
    """
    NDCI = (B5 − B4) / (B5 + B4)
    Rango típico en lagos eutróficos: 0.1 – 0.6
    """
    b5 = _band(image, "red_edge1")
    b4 = _band(image, "red")
    ndci = b5.subtract(b4).divide(b5.add(b4)).rename("NDCI")
    return ndci


def compute_chl_re(image: ee.Image) -> ee.Image:
    """
    CHL_RE = B5 / B4
    Índice simple de relación Borde Rojo / Rojo.
    """
    b5 = _band(image, "red_edge1")
    b4 = _band(image, "red")
    chl_re = b5.divide(b4).rename("CHL_RE")
    return chl_re


def compute_ndwi(image: ee.Image) -> ee.Image:
    """
    NDWI = (B3 − B8) / (B3 + B8)
    Valores > 0 indican agua.
    """
    b3 = _band(image, "green")
    b8 = _band(image, "nir")
    ndwi = b3.subtract(b8).divide(b3.add(b8)).rename("NDWI")
    return ndwi


def compute_fai(image: ee.Image) -> ee.Image:
    """
    FAI = B8 − [B4 + (B11 − B4) × (865 − 665) / (1610 − 665)]
    Detecta algas flotantes y blooms de cianobacterias.
    """
    b4   = _band(image, "red")
    b8   = _band(image, "nir")
    b11  = _band(image, "swir1")
    # Interpolación lineal espectralmente
    slope = (865 - 665) / (1610 - 665)
    baseline = b4.add(b11.subtract(b4).multiply(slope))
    fai = b8.subtract(baseline).rename("FAI")
    return fai


def compute_green_blue(image: ee.Image) -> ee.Image:
    """B3/B2 – ratio verde/azul, proxy de turbidez."""
    b3 = _band(image, "green")
    b2 = _band(image, "blue")
    ratio = b3.divide(b2).rename("B3_B2")
    return ratio


def compute_ndti(image: ee.Image) -> ee.Image:
    """
    NDTI = (B4 − B3) / (B4 + B3)
    Índice de turbidez. Valores más altos → mayor turbidez.
    """
    b4 = _band(image, "red")
    b3 = _band(image, "green")
    ndti = b4.subtract(b3).divide(b4.add(b3)).rename("NDTI")
    return ndti


# ── Aplicar todos los índices a una imagen ────────────────────────────────────
INDEX_FUNCTIONS = {
    "NDCI":  compute_ndci,
    "CHL_RE": compute_chl_re,
    "NDWI":  compute_ndwi,
    "FAI":   compute_fai,
    "B3_B2": compute_green_blue,
    "NDTI":  compute_ndti,
}

def add_all_indices(image: ee.Image) -> ee.Image:
    """Agrega todas las bandas de índices a la imagen."""
    bands = [fn(image) for fn in INDEX_FUNCTIONS.values()]
    return image.addBands(bands)


def compute_selected_index(image: ee.Image, index_name: str) -> ee.Image:
    """
    Calcula un índice específico por nombre.

    Parámetros
    ----------
    image      : ee.Image  compuesto Sentinel-2 escalado
    index_name : str       uno de los keys de INDEX_FUNCTIONS

    Retorna
    -------
    ee.Image con la banda del índice.
    """
    if index_name not in INDEX_FUNCTIONS:
        raise ValueError(
            f"Índice '{index_name}' desconocido. Disponibles: {list(INDEX_FUNCTIONS.keys())}"
        )
    return INDEX_FUNCTIONS[index_name](image)


# ── Estadísticas zonales del AOI ──────────────────────────────────────────────
def zonal_stats(
    index_image: ee.Image,
    aoi: ee.Geometry,
    band_name: str,
    scale: int = 20,
) -> dict:
    """
    Calcula estadísticas zonales del índice dentro del AOI.

    Parámetros
    ----------
    index_image : ee.Image   imagen del índice (1 banda)
    aoi         : ee.Geometry
    band_name   : str        nombre de la banda a reducir
    scale       : int        resolución en metros (20 m para B5)

    Retorna
    -------
    dict con mean, median, std, min, max, count (o None si falló)
    """
    try:
        stats = index_image.select(band_name).reduceRegion(
            reducer=ee.Reducer.mean()
                     .combine(ee.Reducer.median(), sharedInputs=True)
                     .combine(ee.Reducer.stdDev(), sharedInputs=True)
                     .combine(ee.Reducer.min(),    sharedInputs=True)
                     .combine(ee.Reducer.max(),    sharedInputs=True)
                     .combine(ee.Reducer.count(),  sharedInputs=True),
            geometry=aoi,
            scale=scale,
            maxPixels=1e9,
        ).getInfo()
        return {
            "mean":   stats.get(f"{band_name}_mean"),
            "median": stats.get(f"{band_name}_median"),
            "std":    stats.get(f"{band_name}_stdDev"),
            "min":    stats.get(f"{band_name}_min"),
            "max":    stats.get(f"{band_name}_max"),
            "count":  stats.get(f"{band_name}_count"),
        }
    except Exception as exc:
        logger.error("Error en estadísticas zonales (%s): %s", band_name, exc)
        return {k: None for k in ("mean", "median", "std", "min", "max", "count")}
