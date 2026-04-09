"""
modules/satellite_data.py
Extracción de imágenes Sentinel-2 desde GEE para el AOI del Lago San Pablo.

Funciones principales:
  - get_sentinel_collection()  → colección filtrada por AOI, fecha y nubosidad
  - mask_clouds()              → elimina nubes usando QA60
  - apply_scale_factors()      → escala reflectancias a [0, 1]
  - get_weekly_composites()    → genera compuestos semanales (mediana)
"""

import logging
from datetime import date, timedelta
from typing import Optional

import ee

from config import (
    SENTINEL_COLLECTION,
    CLOUD_FILTER_PERCENT,
    BANDS,
    TEMPORAL_AGGREGATION,
)

logger = logging.getLogger(__name__)


# ── Máscara de nubes ──────────────────────────────────────────────────────────
def mask_clouds(image: ee.Image) -> ee.Image:
    """
    Elimina píxeles con nubes y cirros usando la banda QA60 de Sentinel-2.
    Bits 10 y 11 corresponden a nubes opacas y cirros respectivamente.
    """
    qa = image.select(BANDS["qa"])
    cloud_bit_mask  = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
           qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask)


# ── Factores de escala SR ─────────────────────────────────────────────────────
def apply_scale_factors(image: ee.Image) -> ee.Image:
    """
    Aplica el factor de escala de Sentinel-2 SR (÷10000).
    Retorna reflectancias en [0, 1].
    """
    optical_bands = image.select("B.*")
    return image.addBands(optical_bands.multiply(0.0001), overwrite=True)


# ── Colección principal ───────────────────────────────────────────────────────
def get_sentinel_collection(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    cloud_pct: int = CLOUD_FILTER_PERCENT,
) -> ee.ImageCollection:
    """
    Retorna la colección Sentinel-2 SR filtrada por:
      - Área de interés (AOI)
      - Rango de fechas
      - Porcentaje máximo de nubosidad

    Las imágenes ya tienen nubes enmascaradas y reflectancias escaladas.
    """
    collection = (
        ee.ImageCollection(SENTINEL_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(mask_clouds)
        .map(apply_scale_factors)
    )
    count = collection.size().getInfo()
    logger.info(
        "Colección Sentinel-2: %d imágenes entre %s y %s (nub. < %d%%).",
        count, start_date, end_date, cloud_pct,
    )
    return collection


# ── Compuestos semanales ──────────────────────────────────────────────────────
def _date_range_intervals(
    start_date: str,
    end_date: str,
    aggregation: str = TEMPORAL_AGGREGATION,
) -> list[tuple[str, str]]:
    """
    Divide el rango de fechas en intervalos según la agregación elegida.

    Retorna lista de tuplas (start, end) como strings 'YYYY-MM-DD'.
    """
    start = date.fromisoformat(start_date)
    end   = date.fromisoformat(end_date)

    step_days = {
        "weekly":    7,
        "biweekly": 14,
        "monthly":  30,
    }.get(aggregation, 7)

    intervals = []
    current = start
    while current < end:
        interval_end = min(current + timedelta(days=step_days), end)
        intervals.append((current.isoformat(), interval_end.isoformat()))
        current = interval_end
    return intervals


def extract_index_per_image(
    collection: ee.ImageCollection,
    aoi: ee.Geometry,
    index_name: str = "NDCI",
    scale: int = 20,
) -> "pd.DataFrame":
    """
    *** FUNCIÓN RÁPIDA – usa solo UNA llamada getInfo() ***

    Calcula el índice para TODAS las imágenes de la colección en el
    servidor de GEE y recupera los resultados en una sola llamada.

    Retorna pd.DataFrame con columnas:
        date, mean, median, std, min, max, count
    (una fila por imagen disponible)
    """
    import pandas as pd
    from modules.chlorophyll import compute_index_server

    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.median(), sharedInputs=True)
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.min(),    sharedInputs=True)
        .combine(ee.Reducer.max(),    sharedInputs=True)
        .combine(ee.Reducer.count(),  sharedInputs=True)
    )

    def process_image(image):
        idx   = compute_index_server(image, index_name)
        stats = idx.reduceRegion(
            reducer=reducer,
            geometry=aoi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True,          # evita error si hay demasiados píxeles
        )
        return ee.Feature(None, stats).set(
            "date", image.date().format("YYYY-MM-dd"),
        )

    fc      = collection.map(process_image)
    fc_info = fc.getInfo()            # ← UNA SOLA llamada a la API

    records = []
    for feat in fc_info.get("features", []):
        props = feat["properties"]
        records.append({
            "date":   props.get("date"),
            "mean":   props.get(f"{index_name}_mean"),
            "median": props.get(f"{index_name}_median"),
            "std":    props.get(f"{index_name}_stdDev"),
            "min":    props.get(f"{index_name}_min"),
            "max":    props.get(f"{index_name}_max"),
            "count":  props.get(f"{index_name}_count"),
        })

    if not records:
        return pd.DataFrame(columns=["date", "mean", "median", "std", "min", "max", "count"])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["mean"]).sort_values("date").reset_index(drop=True)
    logger.info("extract_index_per_image: %d imágenes con datos para %s.", len(df), index_name)
    return df


def get_periodic_composites(
    collection: ee.ImageCollection,
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    aggregation: str = TEMPORAL_AGGREGATION,
    reducer: str = "median",
) -> list[dict]:
    """
    Genera compuestos periódicos (semanal/quincenal/mensual) usando
    mediana o media de píxeles para cada intervalo de tiempo.

    Retorna una lista de dicts con estructura:
        {
            "period_start": "YYYY-MM-DD",
            "period_end":   "YYYY-MM-DD",
            "image":        ee.Image,
            "n_images":     int,
        }
    """
    intervals = _date_range_intervals(start_date, end_date, aggregation)
    composites = []

    ee_reducer = ee.Reducer.median() if reducer == "median" else ee.Reducer.mean()

    for t_start, t_end in intervals:
        subset = collection.filterDate(t_start, t_end)
        n = subset.size().getInfo()

        if n == 0:
            logger.debug("Sin imágenes para el período %s – %s.", t_start, t_end)
            composites.append({
                "period_start": t_start,
                "period_end":   t_end,
                "image":        None,
                "n_images":     0,
            })
            continue

        composite = subset.reduce(ee_reducer).clip(aoi)
        composites.append({
            "period_start": t_start,
            "period_end":   t_end,
            "image":        composite,
            "n_images":     n,
        })
        logger.debug("Compuesto %s→%s: %d imágenes.", t_start, t_end, n)

    logger.info(
        "Compuestos generados: %d períodos (%s), con imagen: %d.",
        len(composites),
        aggregation,
        sum(1 for c in composites if c["image"] is not None),
    )
    return composites


# ── Información de la colección ───────────────────────────────────────────────
def collection_info(collection: ee.ImageCollection) -> dict:
    """
    Retorna metadatos básicos de la colección para mostrar en la UI.
    """
    try:
        count = collection.size().getInfo()
        dates = (
            collection
            .aggregate_array("system:time_start")
            .map(lambda t: ee.Date(t).format("YYYY-MM-dd"))
            .getInfo()
        )
        return {
            "n_images": count,
            "dates":    sorted(dates),
            "first":    dates[0] if dates else None,
            "last":     dates[-1] if dates else None,
        }
    except Exception as exc:
        logger.error("Error al obtener info de la colección: %s", exc)
        return {"n_images": 0, "dates": [], "first": None, "last": None}
