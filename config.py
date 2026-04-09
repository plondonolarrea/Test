"""
config.py
Configuración global de la aplicación de evaluación de eutrofización
del Lago San Pablo, Ecuador.
"""

# ── Proyecto GEE ───────────────────────────────────────────────────────────────
GEE_PROJECT = "ee-plondonolarrea"          # ← Cambia por tu project ID de GCP
SERVICE_ACCOUNT_EMAIL = ""                   # Opcional: si usas Service Account
SERVICE_ACCOUNT_KEY_FILE = ""               # Ruta al JSON de Service Account

# ── Lago San Pablo – coordenadas aproximadas del polígono ─────────────────────
# Polígono simplificado del lago (lon, lat). Fuente: cartografía IGM Ecuador.
LAKE_COORDINATES = [
    [-78.2600, -0.1700],
    [-78.2100, -0.1700],
    [-78.1900, -0.2050],
    [-78.2000, -0.2400],
    [-78.2400, -0.2500],
    [-78.2700, -0.2300],
    [-78.2800, -0.2000],
    [-78.2600, -0.1700],
]
LAKE_CENTER = [-78.235, -0.210]              # [lon, lat] centro aproximado
LAKE_NAME = "Lago San Pablo"
LAKE_CRS = "EPSG:4326"

# ── Sentinel-2 ────────────────────────────────────────────────────────────────
SENTINEL_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_FILTER_PERCENT = 20                    # % máximo de nubosidad por imagen

# Bandas de Sentinel-2 usadas (10 m y 20 m de resolución)
BANDS = {
    "blue":      "B2",    # 490 nm
    "green":     "B3",    # 560 nm
    "red":       "B4",    # 665 nm
    "red_edge1": "B5",    # 705 nm  ← clave para clorofila
    "red_edge2": "B6",    # 740 nm
    "red_edge3": "B7",    # 783 nm
    "nir":       "B8",    # 842 nm
    "nir_narrow":"B8A",   # 865 nm
    "swir1":     "B11",   # 1610 nm
    "qa":        "QA60",  # máscara de nubes
}

# ── Índices de clorofila disponibles ─────────────────────────────────────────
CHLOROPHYLL_INDICES = {
    "NDCI":    "Normalized Difference Chlorophyll Index  (B5−B4)/(B5+B4)",
    "CHL_RE":  "Chlorophyll Red-Edge ratio               B5/B4",
    "NDWI":    "Normalized Difference Water Index        (B3−B8)/(B3+B8)",
    "FAI":     "Floating Algae Index                     B8−B4−(B11−B4)×((865−665)/(1610−665))",
    "B3_B2":   "Green/Blue ratio  (turbidez proxy)       B3/B2",
}

# ── Serie temporal ────────────────────────────────────────────────────────────
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE   = "2023-12-31"
TEMPORAL_AGGREGATION = "weekly"              # "weekly" | "biweekly" | "monthly"

# ── Exportación ──────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
EXPORT_FORMAT = "CSV"                        # "CSV" | "XLSX"

# ── Interfaz ─────────────────────────────────────────────────────────────────
APP_TITLE = "Evaluación de Eutrofización – Lago San Pablo"
APP_ICON  = "🌊"
MAP_ZOOM  = 13
