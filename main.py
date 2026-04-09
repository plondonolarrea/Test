"""
main.py
Punto de entrada CLI para ejecutar el pipeline de análisis sin interfaz
gráfica. Útil para automatización, scripts y pruebas.

Uso:
    python main.py --start 2023-01-01 --end 2023-12-31 --index NDCI
    python main.py --start 2023-01-01 --end 2023-12-31 --index NDCI --aggregation monthly
    python main.py --help
"""

import argparse
import logging
import sys
from pathlib import Path

from config import (
    GEE_PROJECT,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    TEMPORAL_AGGREGATION,
    CLOUD_FILTER_PERCENT,
    OUTPUT_DIR,
)
from modules.gee_auth import setup_gee, get_gee_status
from modules.lake_boundary import get_default_aoi, aoi_from_shapefile, get_aoi_info
from modules.satellite_data import get_sentinel_collection, get_periodic_composites, collection_info
from modules.time_series import extract_time_series, compute_trend, summary_stats, export_to_csv, export_to_excel
from modules.chlorophyll import INDEX_FUNCTIONS

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(OUTPUT_DIR) / "san_pablo.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── Argparse ──────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline CLI – Evaluación de Eutrofización Lago San Pablo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start",       default=DEFAULT_START_DATE, help="Fecha inicio (YYYY-MM-DD)")
    parser.add_argument("--end",         default=DEFAULT_END_DATE,   help="Fecha fin (YYYY-MM-DD)")
    parser.add_argument("--index",       default="NDCI",
                        choices=list(INDEX_FUNCTIONS.keys()),
                        help="Índice de clorofila a calcular")
    parser.add_argument("--aggregation", default=TEMPORAL_AGGREGATION,
                        choices=["weekly", "biweekly", "monthly"],
                        help="Agregación temporal")
    parser.add_argument("--cloud",       default=CLOUD_FILTER_PERCENT, type=int,
                        help="Porcentaje máximo de nubosidad")
    parser.add_argument("--project",     default=GEE_PROJECT, help="Project ID de GCP")
    parser.add_argument("--auth",        default="saved",
                        choices=["saved", "interactive", "service_account"],
                        help="Modo de autenticación GEE")
    parser.add_argument("--sa-email",    default="", help="Email del Service Account (opcional)")
    parser.add_argument("--sa-key",      default="", help="Ruta al JSON del Service Account (opcional)")
    parser.add_argument("--shapefile",   default=None,
                        help="Ruta a shapefile/GeoJSON del lago (si no, se usa el predefinido)")
    parser.add_argument("--format",      default="csv", choices=["csv", "xlsx"],
                        help="Formato de exportación")
    parser.add_argument("--no-export",   action="store_true",
                        help="No exportar resultados a archivo")
    parser.add_argument("--scale",       default=20, type=int,
                        help="Resolución en metros para reduceRegion (20 para B5)")
    return parser


# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(args: argparse.Namespace) -> int:
    """
    Ejecuta el pipeline completo y retorna 0 si tuvo éxito, 1 si falló.
    """
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 1. Autenticación GEE
    logger.info("=== PASO 1: Autenticación GEE ===")
    ok = setup_gee(
        mode=args.auth,
        project=args.project,
        sa_email=args.sa_email,
        sa_key_file=args.sa_key,
    )
    if not ok:
        logger.error("No se pudo autenticar con GEE. Abortando.")
        return 1

    status = get_gee_status()
    logger.info("Estado GEE: %s", status["message"])

    # 2. AOI
    logger.info("=== PASO 2: Cargando AOI ===")
    if args.shapefile:
        aoi = aoi_from_shapefile(args.shapefile)
        if aoi is None:
            logger.warning("Shapefile no válido. Usando AOI predefinido.")
            aoi = get_default_aoi()
    else:
        aoi = get_default_aoi()

    info = get_aoi_info(aoi)
    logger.info("AOI – área: %.3f km², centroide: (%.5f, %.5f)",
                info.get("area_km2", 0),
                info.get("centroid_lon", 0),
                info.get("centroid_lat", 0))

    # 3. Colección Sentinel-2
    logger.info("=== PASO 3: Colección Sentinel-2 (%s → %s) ===", args.start, args.end)
    collection = get_sentinel_collection(
        aoi,
        start_date=args.start,
        end_date=args.end,
        cloud_pct=args.cloud,
    )
    cinfo = collection_info(collection)
    logger.info("Imágenes encontradas: %d (rango: %s → %s)",
                cinfo["n_images"], cinfo["first"], cinfo["last"])

    if cinfo["n_images"] == 0:
        logger.error("Sin imágenes para el período. Revisa los parámetros.")
        return 1

    # 4. Compuestos periódicos
    logger.info("=== PASO 4: Compuestos %s ===", args.aggregation)
    composites = get_periodic_composites(
        collection, aoi,
        start_date=args.start,
        end_date=args.end,
        aggregation=args.aggregation,
    )
    logger.info("Períodos generados: %d", len(composites))

    # 5. Serie temporal
    logger.info("=== PASO 5: Extracción de índice %s ===", args.index)
    ts_df = extract_time_series(
        composites, aoi,
        index_name=args.index,
        scale=args.scale,
    )

    # 6. Estadísticas y tendencia
    logger.info("=== PASO 6: Estadísticas ===")
    stats = summary_stats(ts_df, value_col="mean")
    trend = compute_trend(ts_df, value_col="mean")

    logger.info("Resultados del índice %s:", args.index)
    for k, v in stats.items():
        logger.info("  %-20s %s", k, v)
    logger.info("Tendencia: %s (pendiente=%.7f, R²=%.4f)",
                trend.get("trend"), trend.get("slope", 0), trend.get("r_squared", 0))

    # 7. Exportación
    if not args.no_export:
        logger.info("=== PASO 7: Exportando resultados ===")
        filename = f"serie_{args.index}_{args.start}_{args.end}"
        if args.format == "csv":
            path = export_to_csv(ts_df, filename + ".csv")
        else:
            path = export_to_excel(ts_df, filename + ".xlsx")
        logger.info("Archivo exportado: %s", path)
    else:
        logger.info("Exportación omitida (--no-export).")

    logger.info("=== Pipeline completado correctamente. ===")
    return 0


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()
