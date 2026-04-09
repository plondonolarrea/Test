"""
app.py
Interfaz principal con Shiny para Python – Evaluación de eutrofización
del Lago San Pablo, Ecuador.

Ejecutar con:
    shiny run app.py --reload

Shiny usa un modelo reactivo: solo re-ejecuta los cálculos cuyas
dependencias han cambiado, lo que lo hace significativamente más
eficiente que Streamlit para análisis pesados con GEE.
"""

import io
import logging
from pathlib import Path

import pandas as pd
from shiny import App, reactive, render, req, ui

from config import (
    APP_TITLE, CHLOROPHYLL_INDICES, DEFAULT_END_DATE,
    DEFAULT_START_DATE, GEE_PROJECT, LAKE_CENTER, MAP_ZOOM,
)
from modules.chlorophyll import INDEX_FUNCTIONS
from modules.correlation import (
    REGRESSION_TYPES,
    best_model,
    compare_all_regressions,
    compute_all_correlations,
    fit_regression,
    get_numeric_field_cols,
    load_field_data,
    merge_satellite_field,
    plot_correlation_heatmap,
    plot_pairwise_scatter,
    plot_regression_comparison,
    plot_residuals,
    plot_scatter_regression,
)
from modules.gee_auth import get_gee_status, setup_gee
from modules.lake_boundary import (
    aoi_from_shapefile,
    aoi_to_geojson,
    get_aoi_info,
    get_default_aoi,
)
from modules.satellite_data import (
    collection_info,
    get_sentinel_collection,
)
from modules.time_series import (
    compute_trend,
    export_to_csv,
    summary_stats,
    extract_time_series_fast,
)
from modules.visualization import (
    build_lake_map,
    classify_eutrophication,
    plot_eutrophication_gauge,
    plot_image_availability,
    plot_monthly_distribution,
    plot_time_series,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

app_ui = ui.page_navbar(

    # Favicon inline (evita el 404 del navegador)
    ui.head_content(
        ui.tags.link(
            rel="icon",
            type="image/svg+xml",
            href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🌊</text></svg>",
        )
    ),

    # ── Tab 1: Mapa ────────────────────────────────────────────────────────────
    ui.nav_panel(
        "🗺️ Mapa del lago",
        ui.output_ui("map_html"),
        ui.output_ui("aoi_metrics_ui"),
    ),

    # ── Tab 2: Serie temporal ──────────────────────────────────────────────────
    ui.nav_panel(
        "📈 Serie temporal",
        ui.output_ui("ts_alert_ui"),
        ui.output_ui("gauge_row_ui"),
        ui.output_ui("fig_ts"),
        ui.accordion(
            ui.accordion_panel(
                "Disponibilidad de imágenes por período",
                ui.output_ui("fig_avail"),
            ),
            ui.accordion_panel(
                "Tabla de datos completa",
                ui.output_data_frame("ts_table"),
            ),
            open=False,
        ),
        ui.output_ui("download_btns_ui"),
    ),

    # ── Tab 3: Estadísticas ────────────────────────────────────────────────────
    ui.nav_panel(
        "📊 Estadísticas",
        ui.output_ui("stats_grid_ui"),
        ui.output_ui("fig_monthly"),
        ui.output_ui("trend_ui"),
    ),

    # ── Tab 4: Correlación ─────────────────────────────────────────────────────
    ui.nav_panel(
        "🔬 Correlación (campo)",
        ui.layout_columns(
            ui.card(
                ui.card_header("1 · Cargar datos de campo"),
                ui.input_file(
                    "field_file",
                    "Archivo CSV o Excel",
                    accept=[".csv", ".xlsx", ".xls"],
                    placeholder="Sin archivo seleccionado",
                ),
                ui.output_ui("field_info_ui"),
                fill=False,
            ),
            ui.card(
                ui.card_header("2 · Configurar cruce temporal"),
                ui.input_slider(
                    "tolerance_days",
                    "Tolerancia de fechas (±días)",
                    min=1, max=21, value=3, step=1,
                ),
                ui.input_action_button(
                    "merge_btn",
                    "Cruzar datos satelital ↔ campo",
                    class_="btn-primary w-100",
                ),
                ui.output_ui("merge_info_ui"),
                fill=False,
            ),
            col_widths=[5, 7],
        ),
        ui.hr(),
        ui.output_ui("corr_tabs_ui"),
    ),

    # ── Barra lateral global ───────────────────────────────────────────────────
    sidebar=ui.sidebar(
        ui.h5("1 · Conexión a GEE", class_="mt-0"),
        ui.input_text("gee_project", "Project ID de GCP", value=GEE_PROJECT),
        ui.input_select(
            "auth_mode",
            "Modo de autenticación",
            {
                "saved":           "Credenciales guardadas (OAuth)",
                "interactive":     "OAuth interactivo (navegador)",
                "service_account": "Service Account (JSON)",
            },
        ),
        ui.output_ui("sa_inputs_ui"),
        ui.input_action_button(
            "connect_gee", "Conectar a GEE", class_="btn-primary w-100 mb-1",
        ),
        ui.output_ui("gee_badge_ui"),
        ui.accordion(
            ui.accordion_panel(
                "ℹ️ ¿No conecta?",
                ui.p("Si usas 'Credenciales guardadas', ejecuta este comando "
                     "en la terminal UNA sola vez:", class_="small mb-1"),
                ui.tags.code("earthengine authenticate",
                             style="font-size:0.8rem; background:#e8f0fe; "
                                   "padding:4px 8px; border-radius:4px; display:block;"),
                ui.p("Luego vuelve a pulsar 'Conectar a GEE'.",
                     class_="small mt-1 mb-0"),
            ),
            open=False,
        ),

        ui.hr(),
        ui.h5("2 · Área de Interés (AOI)"),
        ui.input_radio_buttons(
            "aoi_source",
            None,
            {
                "default": "Predefinido (Lago San Pablo)",
                "upload":  "Subir shapefile / GeoJSON",
            },
        ),
        ui.output_ui("shp_upload_ui"),

        ui.hr(),
        ui.h5("3 · Parámetros de análisis"),
        ui.input_date("start_date", "Fecha inicio", value=DEFAULT_START_DATE),
        ui.input_date("end_date",   "Fecha fin",    value=DEFAULT_END_DATE),
        ui.input_select(
            "index_name",
            "Índice de clorofila",
            {k: f"{k} – {v[:38]}…" for k, v in CHLOROPHYLL_INDICES.items()},
        ),
        ui.input_select(
            "aggregation",
            "Agregación temporal",
            {"weekly": "Semanal (7d)",
             "biweekly": "Quincenal (14d)",
             "monthly": "Mensual (~30d)"},
        ),
        ui.input_slider(
            "cloud_pct",
            "Nubosidad máxima (%)",
            min=5, max=80, value=20, step=5,
        ),
        ui.hr(),
        ui.output_ui("run_btn_ui"),

        width=330,
        bg="#F8F9FA",
    ),

    title=ui.span("🌊 ", ui.strong(APP_TITLE)),
    navbar_options=ui.navbar_options(bg="#1565C0", inverse=True),
    fillable=True,
    id="main_navbar",
)


# ══════════════════════════════════════════════════════════════════════════════
# SERVER
# ══════════════════════════════════════════════════════════════════════════════

def _plotly(fig) -> ui.HTML:
    """Convierte una figura Plotly a HTML embebible en Shiny (sin shinywidgets)."""
    return ui.HTML(fig.to_html(full_html=False, include_plotlyjs="cdn"))


def server(input, output, session):

    # ── Estado reactivo global ─────────────────────────────────────────────────
    gee_ready     = reactive.Value(False)
    aoi_obj       = reactive.Value(None)      # ee.Geometry
    ts_df         = reactive.Value(None)      # pd.DataFrame serie temporal
    field_df      = reactive.Value(None)      # pd.DataFrame datos de campo
    merged_df     = reactive.Value(None)      # pd.DataFrame fusionado

    # ── Inputs dinámicos de la barra lateral ───────────────────────────────────
    @render.ui
    def sa_inputs_ui():
        if input.auth_mode() == "service_account":
            return ui.TagList(
                ui.input_text("sa_email",   "Email del Service Account", ""),
                ui.input_text("sa_key",     "Ruta al JSON de clave",    ""),
            )
        return ui.TagList()

    @render.ui
    def shp_upload_ui():
        if input.aoi_source() == "upload":
            return ui.input_file(
                "shp_file",
                "Shapefile / GeoJSON / ZIP",
                accept=[".shp", ".geojson", ".json", ".zip"],
                placeholder="Selecciona archivo…",
            )
        return ui.TagList()

    @render.ui
    def gee_badge_ui():
        if gee_ready.get():
            return ui.p("● GEE conectado", class_="text-success fw-bold small mb-0")
        return ui.p("● GEE desconectado", class_="text-danger fw-bold small mb-0")

    @render.ui
    def run_btn_ui():
        disabled = not gee_ready.get()
        return ui.input_action_button(
            "run_analysis",
            "▶ Ejecutar análisis",
            class_=f"btn-{'primary' if not disabled else 'secondary'} w-100",
            disabled=disabled,
        )

    # ── Conexión GEE ───────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.connect_gee)
    def _connect_gee():
        try:
            mode = input.auth_mode()

            # Los inputs sa_email / sa_key solo existen en el DOM
            # cuando el modo es "service_account"
            sa_email = ""
            sa_key   = ""
            if mode == "service_account":
                try:
                    sa_email = input.sa_email()
                    sa_key   = input.sa_key()
                except Exception:
                    pass

            ui.notification_show(
                "⏳ Autenticando con GEE…",
                id="gee_notif", duration=None, type="message",
            )

            ok, msg = setup_gee(
                mode=input.auth_mode(),
                project=input.gee_project(),
                sa_email=sa_email,
                sa_key_file=sa_key,
            )

            ui.notification_remove("gee_notif")

            if ok:
                gee_ready.set(True)
                ui.notification_show(
                    f"✅ {msg}", type="message", duration=6,
                )
            else:
                gee_ready.set(False)
                ui.notification_show(
                    f"❌ {msg}", type="error", duration=None,
                )

        except Exception as exc:
            # Captura cualquier excepción inesperada y la muestra en pantalla
            import traceback
            tb = traceback.format_exc()
            logger.exception("Excepción no controlada en _connect_gee")
            try:
                ui.notification_remove("gee_notif")
            except Exception:
                pass
            ui.notification_show(
                f"❌ Excepción inesperada:\n{exc}\n\nDetalle en consola.",
                type="error",
                duration=None,
            )
            gee_ready.set(False)

    # ── AOI reactivo ───────────────────────────────────────────────────────────
    @reactive.calc
    def current_aoi():
        req(gee_ready.get())
        if input.aoi_source() == "upload":
            files = input.shp_file()
            if files:
                aoi = aoi_from_shapefile(files[0]["datapath"])
                if aoi is not None:
                    aoi_obj.set(aoi)
                    return aoi
        aoi = get_default_aoi()
        aoi_obj.set(aoi)
        return aoi

    # ── Pipeline de análisis ───────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.run_analysis)
    def _run_analysis():
        req(gee_ready.get())
        aoi = current_aoi()

        start = input.start_date().isoformat()
        end   = input.end_date().isoformat()
        idx   = input.index_name()
        agg   = input.aggregation()
        cloud = input.cloud_pct()

        ui.notification_show(
            f"⏳ Calculando {idx} en GEE ({start} → {end})… "
            "Esto puede tardar 15-30 segundos.",
            id="analysis_notif", duration=None, type="message",
        )
        try:
            # ── Verificación rápida de imágenes disponibles ────────────────────
            collection = get_sentinel_collection(aoi, start, end, cloud)
            cinfo      = collection_info(collection)

            if cinfo["n_images"] == 0:
                ui.notification_remove("analysis_notif")
                ui.notification_show(
                    f"Sin imágenes Sentinel-2 para el período {start} → {end} "
                    f"con nubosidad < {cloud}%. Amplía el filtro o el rango de fechas.",
                    type="warning", duration=10,
                )
                return

            ui.notification_show(
                f"⏳ {cinfo['n_images']} imágenes encontradas. "
                f"Extrayendo {idx} server-side…",
                id="analysis_notif", duration=None, type="message",
            )

            # ── Pipeline rápido: 1 sola llamada getInfo() ──────────────────────
            result = extract_time_series_fast(
                aoi,
                start_date=start,
                end_date=end,
                index_name=idx,
                aggregation=agg,
                cloud_pct=cloud,
            )

            ui.notification_remove("analysis_notif")

            if result.empty:
                ui.notification_show(
                    "El análisis no devolvió datos. "
                    "Verifica que el AOI coincida con el área del lago "
                    "y que haya imágenes con píxeles válidos.",
                    type="warning", duration=10,
                )
                return

            ts_df.set(result)
            n_data = int(result["n_images"].sum())
            ui.notification_show(
                f"✅ Completado: {len(result)} períodos · {n_data} imágenes procesadas.",
                type="message", duration=6,
            )

        except Exception as exc:
            ui.notification_remove("analysis_notif")
            ui.notification_show(f"❌ Error: {exc}", type="error", duration=None)
            logger.exception("Error en pipeline de análisis")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 – MAPA
    # ══════════════════════════════════════════════════════════════════════════

    @render.ui
    def map_html():
        aoi_geojson = None
        if gee_ready.get():
            try:
                aoi = current_aoi()
                aoi_geojson = aoi_to_geojson(aoi)
            except Exception:
                pass
        m = build_lake_map(aoi_geojson=aoi_geojson)
        return ui.HTML(m._repr_html_())

    @render.ui
    def aoi_metrics_ui():
        if not gee_ready.get():
            return ui.div(
                ui.p("Conecta a GEE para ver la información del AOI.",
                     class_="text-muted mt-2"),
            )
        try:
            aoi  = current_aoi()
            info = get_aoi_info(aoi)
            return ui.layout_columns(
                ui.value_box("Área del AOI",    f"{info.get('area_km2','N/A')} km²",
                             theme="primary"),
                ui.value_box("Centroide Lon",   str(info.get("centroid_lon", "N/A")),
                             theme="secondary"),
                ui.value_box("Centroide Lat",   str(info.get("centroid_lat", "N/A")),
                             theme="secondary"),
                col_widths=[4, 4, 4],
            )
        except Exception:
            return ui.TagList()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 – SERIE TEMPORAL
    # ══════════════════════════════════════════════════════════════════════════

    @render.ui
    def ts_alert_ui():
        df = ts_df.get()
        if df is None:
            return ui.div(
                ui.p("Configura los parámetros en el panel izquierdo y "
                     "presiona ▶ Ejecutar análisis.",
                     class_="text-muted mt-2"),
            )
        n_data = int(df["mean"].notna().sum())
        idx    = input.index_name()
        return ui.div(
            ui.p(
                f"Serie temporal de ",
                ui.strong(idx),
                f" · {n_data} períodos con datos · "
                f"{input.start_date().isoformat()} → {input.end_date().isoformat()}",
                class_="text-muted small mt-1 mb-2",
            )
        )

    @render.ui
    def gauge_row_ui():
        df = ts_df.get()
        if df is None or df["mean"].notna().sum() == 0:
            return ui.TagList()

        last_val = float(df["mean"].dropna().iloc[-1])
        idx      = input.index_name()
        level    = classify_eutrophication(last_val, idx)
        trend    = compute_trend(df, "mean")

        return ui.layout_columns(
            ui.card(
                ui.card_header("Último valor analizado"),
                ui.output_ui("fig_gauge"),
                fill=False,
            ),
            ui.card(
                ui.card_header("Resumen"),
                ui.tags.dl(
                    ui.tags.dt(f"Valor {idx}"),
                    ui.tags.dd(f"{last_val:.5f}"),
                    ui.tags.dt("Nivel de eutrofización"),
                    ui.tags.dd(level),
                    ui.tags.dt("Tendencia OLS"),
                    ui.tags.dd(
                        f"{trend.get('trend','N/A')} "
                        f"(pendiente = {trend.get('slope', 0):.6f}/día, "
                        f"R² = {trend.get('r_squared', 0):.3f})"
                    ),
                ),
                fill=False,
            ),
            col_widths=[5, 7],
        )

    @render.ui
    def fig_gauge():
        df = ts_df.get()
        req(df is not None and df["mean"].notna().any())
        last_val = float(df["mean"].dropna().iloc[-1])
        return _plotly(plot_eutrophication_gauge(last_val, input.index_name()))

    @render.ui
    def fig_ts():
        df = ts_df.get()
        req(df is not None and not df.empty)
        trend = compute_trend(df, "mean")
        return _plotly(plot_time_series(df, input.index_name(), show_trend=True, trend_info=trend))

    @render.ui
    def fig_avail():
        df = ts_df.get()
        req(df is not None)
        return _plotly(plot_image_availability(df))

    @render.data_frame
    def ts_table():
        df = ts_df.get()
        req(df is not None)
        display = df.copy()
        for c in ["mean", "median", "std", "min", "max"]:
            if c in display.columns:
                display[c] = display[c].round(5)
        return render.DataGrid(display, filters=True)

    @render.ui
    def download_btns_ui():
        df = ts_df.get()
        if df is None:
            return ui.TagList()
        return ui.layout_columns(
            ui.download_button("dl_csv",  "⬇ Descargar CSV",   class_="btn-outline-primary w-100"),
            ui.download_button("dl_xlsx", "⬇ Descargar Excel", class_="btn-outline-success w-100"),
            col_widths=[6, 6],
        )

    @render.download(filename=lambda: f"serie_{input.index_name()}.csv")
    def dl_csv():
        df = ts_df.get()
        req(df is not None)
        return df.to_csv(index=False, encoding="utf-8-sig")

    @render.download(filename=lambda: f"serie_{input.index_name()}.xlsx")
    def dl_xlsx():
        df = ts_df.get()
        req(df is not None)
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        return buf.getvalue()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 – ESTADÍSTICAS
    # ══════════════════════════════════════════════════════════════════════════

    @render.ui
    def stats_grid_ui():
        df = ts_df.get()
        if df is None or df.empty:
            return ui.p("Ejecuta el análisis para ver las estadísticas.",
                        class_="text-muted mt-2")

        s = summary_stats(df, "mean")
        metrics = [
            ("Períodos",       "n_períodos"),
            ("Con datos",      "con_datos"),
            ("Media",          "media"),
            ("Mediana",        "mediana"),
            ("Desv. estándar", "desv_estándar"),
            ("Mínimo",         "mínimo"),
            ("Máximo",         "máximo"),
            ("Rango",          "rango"),
        ]
        boxes = [
            ui.value_box(label, str(s.get(key, "N/A")), theme="primary")
            for label, key in metrics
        ]
        return ui.layout_columns(*boxes, col_widths=[3] * 8)

    @render.ui
    def fig_monthly():
        df = ts_df.get()
        req(df is not None and df["mean"].notna().sum() >= 3)
        return _plotly(plot_monthly_distribution(df, "mean"))

    @render.ui
    def trend_ui():
        df = ts_df.get()
        if df is None:
            return ui.TagList()
        trend = compute_trend(df, "mean")
        return ui.card(
            ui.card_header("Análisis de tendencia (regresión OLS)"),
            ui.layout_columns(
                ui.value_box("Tendencia",  trend.get("trend", "N/A"),    theme="secondary"),
                ui.value_box("R²",         f"{trend.get('r_squared', 0):.4f}" if trend.get("r_squared") else "N/A", theme="secondary"),
                ui.value_box("p-valor",    f"{trend.get('p_value', 1):.4f}"   if trend.get("p_value")   else "N/A", theme="secondary"),
                ui.value_box("Pendiente",  f"{trend.get('slope', 0):.6f}/día" if trend.get("slope")     else "N/A", theme="secondary"),
                col_widths=[3, 3, 3, 3],
            ),
            fill=False,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 – CORRELACIÓN
    # ══════════════════════════════════════════════════════════════════════════

    # Carga de datos de campo
    @reactive.effect
    @reactive.event(input.field_file)
    def _load_field():
        files = input.field_file()
        if not files:
            return
        df = load_field_data(files[0]["datapath"])
        field_df.set(df)
        merged_df.set(None)
        if df is not None:
            ui.notification_show(
                f"Datos de campo cargados: {len(df)} registros, "
                f"columnas: {', '.join(df.columns.tolist())}",
                type="message", duration=5,
            )
        else:
            ui.notification_show(
                "No se pudo leer el archivo de campo.", type="error", duration=5,
            )

    @render.ui
    def field_info_ui():
        df = field_df.get()
        if df is None:
            return ui.p("Sin datos cargados.", class_="text-muted small")
        cols = get_numeric_field_cols(df)
        return ui.TagList(
            ui.p(f"✓ {len(df)} registros | Columnas numéricas: {', '.join(cols)}",
                 class_="text-success small mt-1"),
            ui.output_data_frame("field_preview"),
        )

    @render.data_frame
    def field_preview():
        df = field_df.get()
        req(df is not None)
        return render.DataGrid(df.head(8))

    # Cruce temporal
    @reactive.effect
    @reactive.event(input.merge_btn)
    def _do_merge():
        sat = ts_df.get()
        fld = field_df.get()

        if sat is None:
            ui.notification_show(
                "Ejecuta primero el análisis satelital (pestaña Serie temporal).",
                type="warning", duration=5,
            )
            return
        if fld is None:
            ui.notification_show("Carga primero los datos de campo.", type="warning", duration=5)
            return

        result = merge_satellite_field(sat, fld, tolerance_days=input.tolerance_days())
        merged_df.set(result)

        if result.empty:
            ui.notification_show(
                "Sin coincidencias temporales. "
                "Amplía la tolerancia o el rango de fechas del análisis.",
                type="warning", duration=6,
            )
        else:
            ui.notification_show(
                f"Cruce completado: {len(result)} coincidencias encontradas.",
                type="message", duration=4,
            )

    @render.ui
    def merge_info_ui():
        df = merged_df.get()
        if df is None:
            return ui.TagList()
        if df.empty:
            return ui.p("Sin coincidencias.", class_="text-warning small")
        return ui.p(f"✓ {len(df)} coincidencias listas para análisis.",
                    class_="text-success small mt-1")

    # Contenido dinámico de la pestaña correlación
    @render.ui
    def corr_tabs_ui():
        df = merged_df.get()
        if df is None or df.empty:
            return ui.p(
                "Sube datos de campo y cruza con el análisis satelital para "
                "ver los resultados de correlación.",
                class_="text-muted mt-3",
            )

        fld_cols = [c for c in df.columns
                    if c not in ("fecha", "sat_date", "sat_mean", "sat_median", "delta_days")
                    and pd.api.types.is_numeric_dtype(df[c])]

        if not fld_cols:
            return ui.p("No se encontraron columnas numéricas en los datos de campo.",
                        class_="text-warning")

        return ui.navset_tab(
            # Sub-tab A: Matriz de correlación
            ui.nav_panel(
                "Matriz de correlación",
                ui.layout_columns(
                    ui.input_select(
                        "corr_type",
                        "Tipo de correlación",
                        {"pearson_r": "Pearson r", "spearman_r": "Spearman ρ"},
                    ),
                    col_widths=[4],
                ),
                ui.output_ui("fig_heatmap"),
                ui.output_data_frame("corr_table"),
            ),
            # Sub-tab B: Regresión individual
            ui.nav_panel(
                "Regresión individual",
                ui.layout_columns(
                    ui.input_select(
                        "reg_y_col",
                        "Parámetro de campo (Y)",
                        {c: c.upper() for c in fld_cols},
                    ),
                    ui.input_select(
                        "reg_type",
                        "Tipo de regresión",
                        {r: r for r in REGRESSION_TYPES},
                    ),
                    col_widths=[5, 5],
                ),
                ui.output_ui("fig_scatter"),
                ui.output_data_frame("reg_comparison_table"),
                ui.output_ui("fig_residuals"),
            ),
            # Sub-tab C: Vista general (todos los parámetros)
            ui.nav_panel(
                "Todos los parámetros",
                ui.output_ui("fig_pairwise"),
            ),
            # Sub-tab D: Predicción
            ui.nav_panel(
                "Predicción",
                ui.output_ui("prediction_ui"),
            ),
            id="corr_subtabs",
        )

    # ── Sub-tab A: Heatmap de correlación ─────────────────────────────────────
    @reactive.calc
    def corr_matrix():
        df = merged_df.get()
        req(df is not None and not df.empty)
        fld_cols = [c for c in df.columns
                    if c not in ("fecha", "sat_date", "sat_mean", "sat_median", "delta_days")
                    and pd.api.types.is_numeric_dtype(df[c])]
        sat_cols = [c for c in ("sat_mean", "sat_median") if c in df.columns]
        return compute_all_correlations(df, sat_cols, fld_cols)

    @render.ui
    def fig_heatmap():
        cm = corr_matrix()
        req(not cm.empty)
        corr_type = input.corr_type() if hasattr(input, "corr_type") else "pearson_r"
        return _plotly(plot_correlation_heatmap(cm, corr_type=corr_type))

    @render.data_frame
    def corr_table():
        cm = corr_matrix()
        req(not cm.empty)
        return render.DataGrid(cm.round(5), filters=True)

    # ── Sub-tab B: Regresión individual ───────────────────────────────────────
    @reactive.calc
    def current_regression():
        df = merged_df.get()
        req(df is not None and not df.empty)
        y_col    = input.reg_y_col() if hasattr(input, "reg_y_col") else None
        reg_type = input.reg_type()  if hasattr(input, "reg_type")  else "lineal"
        req(y_col and y_col in df.columns)
        x = df["sat_mean"].values.astype(float)
        y = df[y_col].values.astype(float)
        return fit_regression(x, y, reg_type)

    @render.ui
    def fig_scatter():
        df  = merged_df.get()
        req(df is not None and not df.empty)
        y_col = input.reg_y_col() if hasattr(input, "reg_y_col") else None
        req(y_col)
        model = current_regression()
        return _plotly(plot_scatter_regression(df, "sat_mean", y_col, model, input.index_name()))

    @render.data_frame
    def reg_comparison_table():
        df = merged_df.get()
        req(df is not None and not df.empty)
        y_col = input.reg_y_col() if hasattr(input, "reg_y_col") else None
        req(y_col and y_col in df.columns)
        x = df["sat_mean"].values.astype(float)
        y = df[y_col].values.astype(float)
        comp = compare_all_regressions(x, y, input.index_name(), y_col)
        return render.DataGrid(comp.round(5))

    @render.ui
    def fig_residuals():
        df    = merged_df.get()
        req(df is not None and not df.empty)
        y_col = input.reg_y_col() if hasattr(input, "reg_y_col") else None
        req(y_col and y_col in df.columns)
        model = current_regression()
        req(model.get("predict_fn") is not None)
        x = df["sat_mean"].values.astype(float)
        y = df[y_col].values.astype(float)
        return _plotly(plot_residuals(x, y, model, input.index_name(), y_col))

    # ── Sub-tab C: Todos los parámetros ───────────────────────────────────────
    @render.ui
    def fig_pairwise():
        df = merged_df.get()
        req(df is not None and not df.empty)
        fld_cols = [c for c in df.columns
                    if c not in ("fecha", "sat_date", "sat_mean", "sat_median", "delta_days")
                    and pd.api.types.is_numeric_dtype(df[c])]
        req(fld_cols)
        models = {}
        for fc in fld_cols:
            x = df["sat_mean"].values.astype(float)
            y = df[fc].values.astype(float)
            models[fc] = best_model(x, y)
        return _plotly(plot_pairwise_scatter(df, "sat_mean", fld_cols, models))

    # ── Sub-tab D: Predicción ─────────────────────────────────────────────────
    @render.ui
    def prediction_ui():
        df = merged_df.get()
        if df is None or df.empty:
            return ui.p("Sin datos disponibles.", class_="text-muted")

        fld_cols = [c for c in df.columns
                    if c not in ("fecha", "sat_date", "sat_mean", "sat_median", "delta_days")
                    and pd.api.types.is_numeric_dtype(df[c])]
        if not fld_cols:
            return ui.p("Sin columnas numéricas.", class_="text-warning")

        return ui.TagList(
            ui.card(
                ui.card_header("Predicción de parámetros de campo"),
                ui.p(
                    "Introduce un valor del índice satelital para estimar los parámetros "
                    "de campo utilizando el mejor modelo ajustado para cada variable.",
                    class_="text-muted small",
                ),
                ui.input_numeric(
                    "pred_sat_value",
                    f"Valor del índice {input.index_name()}",
                    value=0.1, step=0.001,
                ),
                ui.input_action_button(
                    "predict_btn",
                    "Calcular predicción",
                    class_="btn-primary",
                ),
                ui.output_ui("pred_results_ui"),
                fill=False,
            ),
        )

    @render.ui
    @reactive.event(input.predict_btn)
    def pred_results_ui():
        df = merged_df.get()
        req(df is not None and not df.empty)
        sat_val = input.pred_sat_value()
        fld_cols = [c for c in df.columns
                    if c not in ("fecha", "sat_date", "sat_mean", "sat_median", "delta_days")
                    and pd.api.types.is_numeric_dtype(df[c])]

        boxes = []
        for fc in fld_cols:
            x = df["sat_mean"].values.astype(float)
            y = df[fc].values.astype(float)
            m = best_model(x, y)
            if m.get("predict_fn") is not None:
                pred = float(m["predict_fn"]([sat_val])[0])
                label = f"{fc.upper()} (R²={m.get('r_squared',0):.3f})"
                boxes.append(
                    ui.value_box(label, f"{pred:.4f}", theme="success")
                )
            else:
                boxes.append(
                    ui.value_box(fc.upper(), "N/A – sin modelo", theme="secondary")
                )

        if not boxes:
            return ui.p("Sin modelos disponibles.", class_="text-muted")

        return ui.layout_columns(*boxes, col_widths=[max(2, 12 // len(boxes))] * len(boxes))


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════
app = App(app_ui, server)
