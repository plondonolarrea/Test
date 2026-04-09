"""
modules/visualization.py
Módulo de visualización: gráficos de serie temporal, mapas y
componentes reutilizables para la interfaz Streamlit.
"""

import logging
from typing import Optional

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


# ── Colores por nivel de eutrofización (basado en NDCI) ──────────────────────
EUTROPHICATION_THRESHOLDS = {
    "Oligotrófico":  (-1.0, 0.05),
    "Mesotrófico":   (0.05, 0.15),
    "Eutrófico":     (0.15, 0.30),
    "Hipereutrófico":(0.30,  1.0),
}

LEVEL_COLORS = {
    "Oligotrófico":   "#2196F3",   # azul
    "Mesotrófico":    "#4CAF50",   # verde
    "Eutrófico":      "#FF9800",   # naranja
    "Hipereutrófico": "#F44336",   # rojo
}


def classify_eutrophication(value: float, index: str = "NDCI") -> str:
    """Clasifica el nivel de eutrofización según el valor del índice."""
    if value is None or np.isnan(value):
        return "Sin datos"
    if index == "NDCI":
        for level, (lo, hi) in EUTROPHICATION_THRESHOLDS.items():
            if lo <= value < hi:
                return level
    return "Desconocido"


# ── Gráfico de serie temporal ─────────────────────────────────────────────────
def plot_time_series(
    df: pd.DataFrame,
    index_name: str = "NDCI",
    show_trend: bool = True,
    trend_info: Optional[dict] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Genera un gráfico interactivo de la serie temporal del índice.

    Parámetros
    ----------
    df          : DataFrame con columnas period_start, mean, median, std
    index_name  : nombre del índice para etiquetas
    show_trend  : si True, dibuja la línea de tendencia
    trend_info  : dict con slope, intercept, r_squared (de compute_trend)
    title       : título personalizado

    Retorna
    -------
    plotly.graph_objects.Figure
    """
    df_plot = df.dropna(subset=["mean"]).copy()
    if df_plot.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos disponibles", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    # Banda de error (±1 std)
    fig = go.Figure()

    if "std" in df_plot.columns and df_plot["std"].notna().any():
        fig.add_trace(go.Scatter(
            x=pd.concat([df_plot["period_start"], df_plot["period_start"].iloc[::-1]]),
            y=pd.concat([
                df_plot["mean"] + df_plot["std"].fillna(0),
                (df_plot["mean"] - df_plot["std"].fillna(0)).iloc[::-1],
            ]),
            fill="toself",
            fillcolor="rgba(33, 150, 243, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="±1 Desv. estándar",
        ))

    # Línea principal (media)
    fig.add_trace(go.Scatter(
        x=df_plot["period_start"],
        y=df_plot["mean"],
        mode="lines+markers",
        name=f"{index_name} (media)",
        line=dict(color="#1565C0", width=2),
        marker=dict(size=7, color="#1565C0"),
        hovertemplate=(
            "<b>%{x|%Y-%m-%d}</b><br>"
            f"{index_name}: %{{y:.4f}}<extra></extra>"
        ),
    ))

    # Mediana
    if "median" in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot["period_start"],
            y=df_plot["median"],
            mode="lines",
            name=f"{index_name} (mediana)",
            line=dict(color="#42A5F5", width=1.5, dash="dot"),
        ))

    # Línea de tendencia
    if show_trend and trend_info and trend_info.get("slope") is not None:
        x_days = (df_plot["period_start"] - df_plot["period_start"].min()).dt.days
        y_trend = trend_info["slope"] * x_days + trend_info["intercept"]
        fig.add_trace(go.Scatter(
            x=df_plot["period_start"],
            y=y_trend,
            mode="lines",
            name=f"Tendencia (R²={trend_info['r_squared']:.3f})",
            line=dict(color="#EF5350", width=2, dash="dash"),
        ))

    # Bandas de eutrofización (solo para NDCI)
    if index_name == "NDCI":
        for level, (lo, hi) in EUTROPHICATION_THRESHOLDS.items():
            fig.add_hrect(
                y0=lo, y1=hi,
                fillcolor=LEVEL_COLORS[level],
                opacity=0.06,
                layer="below",
                line_width=0,
                annotation_text=level,
                annotation_position="right",
                annotation=dict(font_size=10, font_color=LEVEL_COLORS[level]),
            )

    _title = title or f"Serie temporal de {index_name} – Lago San Pablo"
    fig.update_layout(
        title=dict(text=_title, font=dict(size=16)),
        xaxis_title="Período",
        yaxis_title=f"Valor {index_name}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white",
        height=480,
    )
    return fig


# ── Gráfico de barras de n_images por período ─────────────────────────────────
def plot_image_availability(df: pd.DataFrame) -> go.Figure:
    """Muestra cuántas imágenes Sentinel-2 hay por período (control de calidad)."""
    fig = px.bar(
        df,
        x="period_start",
        y="n_images",
        labels={"period_start": "Período", "n_images": "Imágenes disponibles"},
        title="Disponibilidad de imágenes Sentinel-2 por período",
        color="n_images",
        color_continuous_scale="Blues",
    )
    fig.update_layout(template="plotly_white", height=300, showlegend=False)
    return fig


# ── Boxplot de distribución por mes ──────────────────────────────────────────
def plot_monthly_distribution(df: pd.DataFrame, value_col: str = "mean") -> go.Figure:
    """Distribución mensual del índice mediante boxplots."""
    df2 = df.dropna(subset=[value_col]).copy()
    df2["mes"] = df2["period_start"].dt.strftime("%b %Y")
    df2["mes_num"] = df2["period_start"].dt.to_period("M").dt.to_timestamp()
    df2 = df2.sort_values("mes_num")

    fig = go.Figure()
    for mes in df2["mes"].unique():
        vals = df2[df2["mes"] == mes][value_col]
        fig.add_trace(go.Box(y=vals, name=mes, boxpoints="all", jitter=0.3))

    fig.update_layout(
        title=f"Distribución mensual de {value_col}",
        yaxis_title=value_col,
        template="plotly_white",
        height=380,
    )
    return fig


# ── Mapa Folium del AOI ───────────────────────────────────────────────────────
def build_lake_map(
    aoi_geojson: Optional[dict] = None,
    center: list = None,
    zoom: int = 13,
) -> folium.Map:
    """
    Genera un mapa Folium con el AOI del lago superpuesto.

    Parámetros
    ----------
    aoi_geojson : dict GeoJSON del AOI (ee.Geometry.getInfo())
    center      : [lat, lon] del centro del mapa
    zoom        : nivel de zoom inicial
    """
    from config import LAKE_CENTER, MAP_ZOOM

    _center = center or [LAKE_CENTER[1], LAKE_CENTER[0]]  # [lat, lon]
    _zoom   = zoom or MAP_ZOOM

    m = folium.Map(
        location=_center,
        zoom_start=_zoom,
        tiles=None,
    )

    # Capa base satélite (Esri)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satélite (Esri)",
        overlay=False,
        control=True,
    ).add_to(m)

    # Capa OSM
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    # AOI del lago
    if aoi_geojson:
        folium.GeoJson(
            data=aoi_geojson,
            name="AOI – Lago San Pablo",
            style_function=lambda _: {
                "fillColor":   "#1565C0",
                "color":       "#0D47A1",
                "weight":      2,
                "fillOpacity": 0.25,
            },
            tooltip="Lago San Pablo",
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


# ── Gauge de nivel de eutrofización ──────────────────────────────────────────
def plot_eutrophication_gauge(value: float, index_name: str = "NDCI") -> go.Figure:
    """
    Velocímetro (gauge) que muestra el nivel actual de eutrofización
    basado en el último valor disponible del índice.
    """
    level = classify_eutrophication(value, index_name)
    color = LEVEL_COLORS.get(level, "#9E9E9E")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(value, 4) if value is not None else 0,
        title={"text": f"Nivel de Eutrofización<br><sub>{level}</sub>", "font": {"size": 16}},
        gauge={
            "axis":  {"range": [-0.2, 0.6], "tickwidth": 1},
            "bar":   {"color": color},
            "steps": [
                {"range": [-1.0, 0.05], "color": "#BBDEFB"},
                {"range": [0.05, 0.15], "color": "#C8E6C9"},
                {"range": [0.15, 0.30], "color": "#FFE0B2"},
                {"range": [0.30,  1.0], "color": "#FFCDD2"},
            ],
            "threshold": {
                "line":  {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": value or 0,
            },
        },
        number={"suffix": f" ({index_name})", "font": {"size": 22}},
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=20, l=30, r=30))
    return fig
