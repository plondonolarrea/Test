"""
modules/lake_boundary.py
Define el Área de Interés (AOI) del Lago San Pablo.

Prioridad de fuentes:
  1. Shapefile cargado por el usuario  (.shp, .geojson, .zip con shp)
  2. Polígono predefinido en config.py (coordenadas IGM Ecuador)

También aplica una máscara de agua usando NDWI para asegurar que el
análisis se restrinja al espejo de agua del lago.
"""

import io
import json
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import ee
import geopandas as gpd
import shapely.geometry

from config import LAKE_COORDINATES, LAKE_CENTER, LAKE_NAME, LAKE_CRS, BANDS

logger = logging.getLogger(__name__)


# ── AOI desde coordenadas predefinidas ────────────────────────────────────────
def get_default_aoi() -> ee.Geometry:
    """
    Retorna la geometría del Lago San Pablo definida en config.py.
    Coordenadas en WGS-84 (EPSG:4326).
    """
    geometry = ee.Geometry.Polygon(LAKE_COORDINATES)
    logger.info("AOI cargado desde coordenadas predefinidas (%s).", LAKE_NAME)
    return geometry


# ── AOI desde shapefile ───────────────────────────────────────────────────────
def aoi_from_shapefile(file_path: str) -> Optional[ee.Geometry]:
    """
    Carga un shapefile (.shp, .geojson o .zip con .shp) y lo convierte
    a ee.Geometry.

    Parámetros
    ----------
    file_path : str
        Ruta local al archivo shapefile o GeoJSON.

    Retorna
    -------
    ee.Geometry o None si falló la carga.
    """
    try:
        path = Path(file_path)

        if path.suffix.lower() == ".zip":
            gdf = _load_shp_from_zip(path)
        elif path.suffix.lower() in {".shp", ".geojson", ".json"}:
            gdf = gpd.read_file(str(path))
        else:
            logger.error("Formato no soportado: %s", path.suffix)
            return None

        # Reproyectar a WGS-84 si es necesario
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        # Unir todas las geometrías en una sola
        union = gdf.geometry.unary_union
        geojson_dict = shapely.geometry.mapping(union)
        aoi = ee.Geometry(geojson_dict)
        logger.info("AOI cargado desde shapefile: %s (registros: %d)", path.name, len(gdf))
        return aoi

    except Exception as exc:
        logger.error("Error al cargar shapefile: %s", exc)
        return None


def aoi_from_bytes(file_bytes: bytes, file_name: str) -> Optional[ee.Geometry]:
    """
    Carga un shapefile/GeoJSON desde bytes (compatible con st.file_uploader).

    Parámetros
    ----------
    file_bytes : bytes
        Contenido del archivo subido.
    file_name : str
        Nombre original del archivo (se usa para detectar extensión).
    """
    suffix = Path(file_name).suffix.lower()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / file_name
            tmp_path.write_bytes(file_bytes)
            return aoi_from_shapefile(str(tmp_path))
    except Exception as exc:
        logger.error("Error al procesar bytes del shapefile: %s", exc)
        return None


def _load_shp_from_zip(zip_path: Path) -> gpd.GeoDataFrame:
    """Extrae un .zip que contiene un shapefile y lo carga con geopandas."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)
        shp_files = list(Path(tmpdir).rglob("*.shp"))
        if not shp_files:
            raise FileNotFoundError("No se encontró archivo .shp dentro del ZIP.")
        return gpd.read_file(str(shp_files[0]))


# ── Máscara de agua (NDWI) ────────────────────────────────────────────────────
def apply_water_mask(image: ee.Image, aoi: ee.Geometry) -> ee.Image:
    """
    Aplica una máscara de agua sobre la imagen usando NDWI (Gao 1996).
    Solo se conservan los píxeles donde NDWI > 0 (agua).

    NDWI = (Green − NIR) / (Green + NIR) = (B3 − B8) / (B3 + B8)
    """
    green = image.select(BANDS["green"])
    nir   = image.select(BANDS["nir"])
    ndwi  = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
    water_mask = ndwi.gt(0)
    return image.updateMask(water_mask)


def get_dynamic_water_mask(aoi: ee.Geometry, year: int = 2022) -> ee.Image:
    """
    Obtiene una máscara de agua permanente desde JRC Global Surface Water
    (más precisa que NDWI para definir el espejo de agua del lago).

    Retorna una imagen binaria: 1 = agua, 0 = tierra.
    """
    jrc = (
        ee.ImageCollection("JRC/GSW1_4/MonthlyHistory")
        .filter(ee.Filter.calendarRange(year, year, "year"))
        .map(lambda img: img.eq(2))          # 2 = agua permanente
        .max()
        .clip(aoi)
    )
    return jrc.rename("water_mask")


# ── Información del AOI ───────────────────────────────────────────────────────
def get_aoi_info(aoi: ee.Geometry) -> dict:
    """
    Retorna metadatos del AOI: área en km², bounds, centroide.
    """
    try:
        area_m2   = aoi.area().getInfo()
        centroid  = aoi.centroid().coordinates().getInfo()
        bounds    = aoi.bounds().coordinates().getInfo()
        return {
            "area_km2": round(area_m2 / 1e6, 3),
            "centroid_lon": round(centroid[0], 5),
            "centroid_lat": round(centroid[1], 5),
            "bounds": bounds,
        }
    except Exception as exc:
        logger.warning("No se pudo obtener info del AOI: %s", exc)
        return {}


def aoi_to_geojson(aoi: ee.Geometry) -> dict:
    """Convierte ee.Geometry a dict GeoJSON (para renderizar en mapas)."""
    return aoi.getInfo()
