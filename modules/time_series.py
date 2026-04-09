"""
modules/time_series.py
Construye la serie temporal del índice de clorofila (u otro índice)
extrayendo estadísticas zonales de cada compuesto periódico.

Flujo:
  composites (list[dict])  →  extract_time_series()  →  pd.DataFrame
  pd.DataFrame             →  export_to_csv() / export_to_excel()
"""

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from config import OUTPUT_DIR
from modules.chlorophyll import compute_selected_index, zonal_stats

logger = logging.getLogger(__name__)


# ── Extracción RÁPIDA (vía server-side GEE + resample pandas) ────────────────
def extract_time_series_fast(
    aoi,
    start_date: str,
    end_date: str,
    index_name: str = "NDCI",
    aggregation: str = "weekly",
    cloud_pct: int = 20,
    scale: int = 20,
) -> pd.DataFrame:
    """
    Pipeline optimizado: una sola llamada getInfo() para toda la serie.

    1. Filtra la colección Sentinel-2 en GEE (server-side).
    2. Calcula el índice en cada imagen dentro de GEE.
    3. Recupera todos los resultados con UNA llamada API.
    4. Agrega a períodos (weekly / biweekly / monthly) con pandas.

    Retorna pd.DataFrame con columnas:
        period_start, period_end, n_images, mean, median, std, min, max
    """
    from modules.satellite_data import get_sentinel_collection, extract_index_per_image

    # 1. Colección filtrada
    collection = get_sentinel_collection(aoi, start_date, end_date, cloud_pct)

    # 2. Extracción server-side (1 llamada API)
    per_image_df = extract_index_per_image(collection, aoi, index_name, scale)

    if per_image_df.empty:
        logger.warning("Sin imágenes con datos válidos para %s.", index_name)
        return pd.DataFrame(
            columns=["period_start", "period_end", "n_images",
                     "mean", "median", "std", "min", "max"]
        )

    # 3. Resamplear a períodos con pandas
    freq_map = {"weekly": "W", "biweekly": "2W", "monthly": "MS"}
    freq = freq_map.get(aggregation, "W")

    df_idx = per_image_df.set_index("date")

    agg = df_idx[["mean", "median", "std", "min", "max"]].resample(freq).agg(
        mean=("mean",   "mean"),
        median=("median", "mean"),
        std=("std",    "mean"),
        min=("min",    "min"),
        max=("max",    "max"),
    )
    agg["n_images"] = df_idx["mean"].resample(freq).count()
    agg = agg.reset_index().rename(columns={"date": "period_end"})

    # Calcular period_start según la frecuencia
    offset_map = {"W": pd.Timedelta(days=7),
                  "2W": pd.Timedelta(days=14),
                  "MS": pd.Timedelta(days=28)}
    offset = offset_map.get(freq, pd.Timedelta(days=7))
    agg["period_start"] = agg["period_end"] - offset

    # Reordenar columnas y descartar períodos sin datos
    result = agg[["period_start", "period_end", "n_images",
                  "mean", "median", "std", "min", "max"]]
    result = result[result["n_images"] > 0].reset_index(drop=True)

    logger.info(
        "Serie temporal (%s): %d períodos con datos.", aggregation, len(result)
    )
    return result


# ── Extracción original (iterativa, conservada para CLI) ──────────────────────
# ── Extracción principal ──────────────────────────────────────────────────────
def extract_time_series(
    composites: list[dict],
    aoi,
    index_name: str = "NDCI",
    scale: int = 20,
) -> pd.DataFrame:
    """
    Itera sobre los compuestos periódicos y extrae estadísticas zonales
    del índice especificado para el AOI del lago.

    Parámetros
    ----------
    composites  : list[dict]  salida de satellite_data.get_periodic_composites()
    aoi         : ee.Geometry
    index_name  : str         índice a calcular (NDCI, CHL_RE, NDWI, …)
    scale       : int         resolución en metros para reduceRegion

    Retorna
    -------
    pd.DataFrame con columnas:
        period_start, period_end, n_images,
        mean, median, std, min, max, count
    """
    records = []

    total = len(composites)
    for i, comp in enumerate(composites):
        logger.debug("Procesando período %d/%d: %s", i + 1, total, comp["period_start"])

        row = {
            "period_start": comp["period_start"],
            "period_end":   comp["period_end"],
            "n_images":     comp["n_images"],
            "mean":   None,
            "median": None,
            "std":    None,
            "min":    None,
            "max":    None,
            "count":  None,
        }

        if comp["image"] is None:
            records.append(row)
            continue

        try:
            index_img = compute_selected_index(comp["image"], index_name)
            stats = zonal_stats(index_img, aoi, band_name=index_name, scale=scale)
            row.update(stats)
        except Exception as exc:
            logger.warning(
                "Error al calcular %s para %s: %s",
                index_name, comp["period_start"], exc,
            )

        records.append(row)

    df = pd.DataFrame(records)
    df["period_start"] = pd.to_datetime(df["period_start"])
    df["period_end"]   = pd.to_datetime(df["period_end"])
    df = df.sort_values("period_start").reset_index(drop=True)

    logger.info(
        "Serie temporal extraída: %d períodos, %d con datos.",
        len(df),
        df["mean"].notna().sum(),
    )
    return df


# ── Tendencia (regresión lineal simple) ───────────────────────────────────────
def compute_trend(df: pd.DataFrame, value_col: str = "mean") -> dict:
    """
    Calcula la tendencia lineal (OLS) de la serie temporal.

    Retorna dict con:
      slope      : pendiente (unidades de índice / día)
      intercept  : intercepto
      r_squared  : coeficiente de determinación R²
      trend      : "ascendente" | "descendente" | "estable"
    """
    from scipy import stats as sp_stats

    clean = df[["period_start", value_col]].dropna()
    if len(clean) < 2:
        return {"slope": None, "intercept": None, "r_squared": None, "trend": "sin datos"}

    x = (clean["period_start"] - clean["period_start"].min()).dt.days.values
    y = clean[value_col].values

    slope, intercept, r, p, se = sp_stats.linregress(x, y)

    trend = "estable"
    if slope > 0.0001:
        trend = "ascendente"
    elif slope < -0.0001:
        trend = "descendente"

    return {
        "slope":     slope,
        "intercept": intercept,
        "r_squared": r ** 2,
        "p_value":   p,
        "trend":     trend,
    }


# ── Exportación ───────────────────────────────────────────────────────────────
def _ensure_output_dir() -> Path:
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    return out


def export_to_csv(df: pd.DataFrame, filename: str = "serie_clorofila.csv") -> str:
    """Guarda el DataFrame como CSV y retorna la ruta."""
    path = _ensure_output_dir() / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("CSV exportado: %s", path)
    return str(path)


def export_to_excel(df: pd.DataFrame, filename: str = "serie_clorofila.xlsx") -> str:
    """Guarda el DataFrame como Excel y retorna la ruta."""
    path = _ensure_output_dir() / filename
    df.to_excel(path, index=False, engine="openpyxl")
    logger.info("Excel exportado: %s", path)
    return str(path)


# ── Resumen estadístico ───────────────────────────────────────────────────────
def summary_stats(df: pd.DataFrame, value_col: str = "mean") -> dict:
    """Retorna estadísticas descriptivas de la serie temporal."""
    s = df[value_col].dropna()
    if s.empty:
        return {}
    return {
        "n_períodos":       len(df),
        "con_datos":        int(s.count()),
        "media":            round(float(s.mean()), 5),
        "mediana":          round(float(s.median()), 5),
        "desv_estándar":    round(float(s.std()), 5),
        "mínimo":           round(float(s.min()), 5),
        "máximo":           round(float(s.max()), 5),
        "rango":            round(float(s.max() - s.min()), 5),
    }
