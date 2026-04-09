"""
modules/correlation.py
Módulo completo de correlación entre índices satelitales y parámetros
físico-químicos medidos en campo (turbidez, DQO, DBO, etc.).

Funcionalidades
───────────────
  · Carga y normalización de datos de campo (CSV / Excel)
  · Fusión temporal satelital ↔ campo con tolerancia configurable
  · Correlación Pearson y Spearman con p-valores e intervalos de confianza
  · Matriz de correlación completa: todos los índices vs todos los parámetros
  · Ajuste y comparación de 4 tipos de regresión por par de variables
  · Selección automática del mejor modelo (máximo R²)
  · Predicción de parámetros de campo a partir de índices satelitales
  · Visualizaciones: heatmap, dispersión con regresión, comparación de modelos
  · Exportación del modelo entrenado (joblib)
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

REGRESSION_TYPES = ["lineal", "logarítmica", "polinómica (grado 2)", "polinómica (grado 3)"]


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS DE CAMPO
# ══════════════════════════════════════════════════════════════════════════════

def load_field_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Carga un CSV o Excel con mediciones de campo.

    Columnas esperadas (mínimas):
        fecha / date  : YYYY-MM-DD
        [parámetros]  : turbidez (NTU), dqo (mg/L), dbo (mg/L), …

    Normaliza nombres de columnas a minúsculas sin espacios.
    """
    path = Path(file_path)
    try:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            logger.error("Formato no soportado: %s", path.suffix)
            return None

        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        date_col = next((c for c in df.columns if "fecha" in c or "date" in c), None)
        if date_col:
            df["fecha"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
            if date_col != "fecha":
                df.drop(columns=[date_col], inplace=True)
        else:
            logger.warning("No se encontró columna de fecha en los datos de campo.")

        logger.info("Datos de campo cargados: %d registros | columnas: %s",
                    len(df), list(df.columns))
        return df
    except Exception as exc:
        logger.error("Error al cargar datos de campo: %s", exc)
        return None


def load_field_data_from_bytes(file_bytes: bytes, file_name: str) -> Optional[pd.DataFrame]:
    """Wrapper para cargar datos desde bytes (subida de archivo en UI)."""
    suffix = Path(file_name).suffix.lower()
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            return load_field_data(tmp.name)
    except Exception as exc:
        logger.error("Error al procesar bytes: %s", exc)
        return None


def get_numeric_field_cols(field_df: pd.DataFrame) -> list[str]:
    """Retorna las columnas numéricas de campo (excluye la columna fecha)."""
    exclude = {"fecha", "date"}
    return [
        c for c in field_df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(field_df[c])
    ]


# ══════════════════════════════════════════════════════════════════════════════
# FUSIÓN TEMPORAL SATELITAL ↔ CAMPO
# ══════════════════════════════════════════════════════════════════════════════

def merge_satellite_field(
    satellite_df: pd.DataFrame,
    field_df: pd.DataFrame,
    tolerance_days: int = 3,
) -> pd.DataFrame:
    """
    Cruza la serie temporal satelital con los datos de campo.

    Para cada medición de campo, busca el compuesto satelital más cercano
    dentro de ±tolerance_days días.

    Retorna DataFrame con columnas:
        fecha, [parámetros campo], sat_date, sat_mean, sat_median, delta_days
    """
    if field_df is None or field_df.empty or "fecha" not in field_df.columns:
        logger.warning("Datos de campo vacíos o sin columna 'fecha'.")
        return pd.DataFrame()

    sat_df = satellite_df.dropna(subset=["mean"]).copy()
    records = []

    for _, row in field_df.iterrows():
        fecha = pd.Timestamp(row["fecha"])
        window = sat_df[
            (sat_df["period_start"] - fecha).abs() <= pd.Timedelta(days=tolerance_days)
        ].copy()

        if window.empty:
            continue

        window["_delta"] = (window["period_start"] - fecha).abs()
        closest = window.nsmallest(1, "_delta").iloc[0]

        rec = row.to_dict()
        rec["sat_date"]   = closest["period_start"]
        rec["sat_mean"]   = closest["mean"]
        rec["sat_median"] = closest.get("median")
        rec["delta_days"] = int(closest["_delta"].days)
        records.append(rec)

    df = pd.DataFrame(records)
    logger.info("Fusión satelital-campo: %d coincidencias de %d registros de campo.",
                len(df), len(field_df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CORRELACIÓN COMPLETA
# ══════════════════════════════════════════════════════════════════════════════

def compute_pearson_spearman(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Calcula Pearson y Spearman entre dos vectores.

    Retorna
    -------
    dict con: pearson_r, pearson_p, spearman_r, spearman_p,
              n, pearson_ci_low, pearson_ci_high
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x_c, y_c = x[mask], y[mask]
    n = len(x_c)

    if n < 3:
        return {k: None for k in (
            "pearson_r", "pearson_p", "spearman_r", "spearman_p",
            "n", "pearson_ci_low", "pearson_ci_high"
        )}

    pr, pp = sp_stats.pearsonr(x_c, y_c)
    sr, sp = sp_stats.spearmanr(x_c, y_c)

    # Intervalo de confianza 95% para Pearson (transformación Fisher)
    z = np.arctanh(pr)
    se = 1 / np.sqrt(n - 3)
    z_ci = sp_stats.norm.ppf(0.975)
    ci_low  = float(np.tanh(z - z_ci * se))
    ci_high = float(np.tanh(z + z_ci * se))

    return {
        "n":              n,
        "pearson_r":      round(float(pr), 4),
        "pearson_p":      round(float(pp), 5),
        "pearson_ci_low": round(ci_low, 4),
        "pearson_ci_high":round(ci_high, 4),
        "spearman_r":     round(float(sr), 4),
        "spearman_p":     round(float(sp), 5),
    }


def compute_all_correlations(
    merged_df: pd.DataFrame,
    sat_cols: list[str],
    field_cols: list[str],
) -> pd.DataFrame:
    """
    Calcula Pearson y Spearman para todas las combinaciones de
    índices satelitales × parámetros de campo.

    Retorna
    -------
    pd.DataFrame con columnas:
        sat_var, field_var, n,
        pearson_r, pearson_p, pearson_sig,
        spearman_r, spearman_p, spearman_sig
    """
    rows = []
    for s_col in sat_cols:
        for f_col in field_cols:
            if s_col not in merged_df.columns or f_col not in merged_df.columns:
                continue
            x = merged_df[s_col].values.astype(float)
            y = merged_df[f_col].values.astype(float)
            stats = compute_pearson_spearman(x, y)
            rows.append({
                "índice_sat":   s_col,
                "parámetro":    f_col,
                "n":            stats["n"],
                "pearson_r":    stats["pearson_r"],
                "pearson_p":    stats["pearson_p"],
                "pearson_sig":  _sig_stars(stats["pearson_p"]),
                "IC95_low":     stats["pearson_ci_low"],
                "IC95_high":    stats["pearson_ci_high"],
                "spearman_r":   stats["spearman_r"],
                "spearman_p":   stats["spearman_p"],
                "spearman_sig": _sig_stars(stats["spearman_p"]),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("pearson_r", key=abs, ascending=False).reset_index(drop=True)
    return df


def _sig_stars(p: Optional[float]) -> str:
    """Convierte p-valor en notación de estrellas de significancia."""
    if p is None:
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ══════════════════════════════════════════════════════════════════════════════
# REGRESIÓN INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════════════

def fit_regression(
    x: np.ndarray,
    y: np.ndarray,
    reg_type: str = "lineal",
) -> dict:
    """
    Ajusta una regresión entre el índice satelital (x) y el parámetro de campo (y).

    Tipos soportados: lineal, logarítmica, polinómica (grado 2), polinómica (grado 3)

    Retorna
    -------
    dict con: model_type, coefs, r_squared, rmse, mae, predict_fn, formula_str
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x_c, y_c = x[mask].astype(float), y[mask].astype(float)

    if len(x_c) < 3:
        return {"r_squared": None, "rmse": None, "mae": None,
                "formula_str": "Insuficientes datos (< 3 puntos)", "predict_fn": None}

    result = {}

    if reg_type == "lineal":
        slope, intercept, r, _, _ = sp_stats.linregress(x_c, y_c)
        predict = lambda xi, s=slope, i=intercept: s * np.asarray(xi) + i
        y_hat = predict(x_c)
        result = {
            "model_type":  "lineal",
            "coefs":       [slope, intercept],
            "formula_str": f"y = {slope:.4f}·x + {intercept:.4f}",
            "r_squared":   round(r ** 2, 5),
            "predict_fn":  predict,
        }

    elif reg_type == "logarítmica":
        x_log = np.log(np.abs(x_c) + 1e-9)
        slope, intercept, r, _, _ = sp_stats.linregress(x_log, y_c)
        predict = lambda xi, s=slope, i=intercept: s * np.log(np.abs(np.asarray(xi)) + 1e-9) + i
        y_hat = predict(x_c)
        result = {
            "model_type":  "logarítmica",
            "coefs":       [slope, intercept],
            "formula_str": f"y = {slope:.4f}·ln(x) + {intercept:.4f}",
            "r_squared":   round(r ** 2, 5),
            "predict_fn":  predict,
        }

    elif "polinómica" in reg_type:
        degree = 3 if "3" in reg_type else 2
        coefs = np.polyfit(x_c, y_c, degree)
        poly  = np.poly1d(coefs)
        y_hat = poly(x_c)
        ss_res = np.sum((y_c - y_hat) ** 2)
        ss_tot = np.sum((y_c - y_c.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
        terms = " + ".join(
            f"{c:.4f}·x^{degree-i}" if (degree-i) > 1
            else (f"{c:.4f}·x" if (degree-i) == 1 else f"{c:.4f}")
            for i, c in enumerate(coefs)
        )
        result = {
            "model_type":  f"polinómica grado {degree}",
            "coefs":       coefs.tolist(),
            "formula_str": f"y = {terms}",
            "r_squared":   round(r2, 5),
            "predict_fn":  lambda xi, p=poly: p(np.asarray(xi)),
        }

    else:
        return {"r_squared": None, "formula_str": f"Tipo desconocido: {reg_type}", "predict_fn": None}

    # Métricas adicionales
    y_hat = result["predict_fn"](x_c)
    result["rmse"] = round(float(np.sqrt(np.mean((y_c - y_hat) ** 2))), 5)
    result["mae"]  = round(float(np.mean(np.abs(y_c - y_hat))), 5)
    result["n"]    = int(len(x_c))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# COMPARACIÓN DE TODOS LOS MODELOS
# ══════════════════════════════════════════════════════════════════════════════

def compare_all_regressions(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str = "índice_sat",
    y_label: str = "parámetro",
) -> pd.DataFrame:
    """
    Ajusta los 4 tipos de regresión para un par (x, y) y retorna
    una tabla comparativa ordenada por R² descendente.

    Útil para seleccionar automáticamente el mejor modelo.
    """
    rows = []
    for reg_type in REGRESSION_TYPES:
        m = fit_regression(x, y, reg_type)
        rows.append({
            "tipo_regresión": reg_type,
            "R²":             m.get("r_squared"),
            "RMSE":           m.get("rmse"),
            "MAE":            m.get("mae"),
            "n":              m.get("n"),
            "fórmula":        m.get("formula_str"),
        })
    df = pd.DataFrame(rows)
    if not df.empty and df["R²"].notna().any():
        df = df.sort_values("R²", ascending=False).reset_index(drop=True)
    return df


def best_model(x: np.ndarray, y: np.ndarray) -> dict:
    """Retorna el modelo con mayor R² entre los 4 tipos de regresión."""
    best = {"r_squared": -1}
    for reg_type in REGRESSION_TYPES:
        m = fit_regression(x, y, reg_type)
        if m.get("r_squared") is not None and m["r_squared"] > best["r_squared"]:
            best = m
    return best


# ══════════════════════════════════════════════════════════════════════════════
# PREDICCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def predict_parameter(model: dict, satellite_values: np.ndarray) -> np.ndarray:
    """
    Aplica el modelo ajustado a nuevos valores satelitales para predecir
    el parámetro de campo.

    Parámetros
    ----------
    model            : dict resultado de fit_regression()
    satellite_values : array de valores del índice satelital

    Retorna
    -------
    np.ndarray de predicciones
    """
    fn = model.get("predict_fn")
    if fn is None:
        raise ValueError("El modelo no tiene función de predicción (predict_fn).")
    return np.asarray(fn(np.asarray(satellite_values, dtype=float)))


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    corr_type: str = "pearson_r",
    title: str = "Matriz de correlación",
) -> go.Figure:
    """
    Heatmap de correlaciones entre índices satelitales y parámetros de campo.

    Parámetros
    ----------
    corr_df   : salida de compute_all_correlations()
    corr_type : "pearson_r" | "spearman_r"
    """
    if corr_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos para graficar", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    pivot = corr_df.pivot(index="índice_sat", columns="parámetro", values=corr_type)
    p_col = "pearson_p" if corr_type == "pearson_r" else "spearman_p"
    pivot_p = corr_df.pivot(index="índice_sat", columns="parámetro", values=p_col)

    # Texto en cada celda: valor + estrellas
    text_matrix = pivot.copy().astype(str)
    for row_idx in pivot.index:
        for col_idx in pivot.columns:
            v = pivot.loc[row_idx, col_idx]
            p = pivot_p.loc[row_idx, col_idx]
            stars = _sig_stars(p) if p is not None else ""
            text_matrix.loc[row_idx, col_idx] = f"{v:.3f}{stars}" if v is not None else "N/A"

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        text=text_matrix.values,
        texttemplate="%{text}",
        colorscale="RdBu",
        zmid=0,
        zmin=-1, zmax=1,
        colorbar=dict(title=corr_type.replace("_", " ").title()),
        hovertemplate=(
            "Índice: %{y}<br>Parámetro: %{x}<br>"
            + corr_type + ": %{z:.4f}<extra></extra>"
        ),
    ))

    _type_label = "Pearson r" if corr_type == "pearson_r" else "Spearman ρ"
    fig.update_layout(
        title=f"{title} ({_type_label})<br><sub>* p<0.05  ** p<0.01  *** p<0.001  ns no significativo</sub>",
        xaxis_title="Parámetro de campo",
        yaxis_title="Índice satelital",
        template="plotly_white",
        height=max(300, 80 * len(pivot)),
    )
    return fig


def plot_scatter_regression(
    merged_df: pd.DataFrame,
    x_col: str = "sat_mean",
    y_col: str = "turbidez",
    model: Optional[dict] = None,
    index_name: str = "NDCI",
) -> go.Figure:
    """Dispersión (x=índice satelital, y=parámetro campo) con curva de regresión."""
    if merged_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos fusionados", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    df_clean = merged_df[[x_col, y_col, "fecha"]].dropna(subset=[x_col, y_col])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_clean[x_col],
        y=df_clean[y_col],
        mode="markers",
        name="Observaciones",
        marker=dict(size=10, color="#1565C0", opacity=0.85,
                    line=dict(width=1, color="white")),
        text=df_clean["fecha"].dt.strftime("%Y-%m-%d"),
        hovertemplate="<b>%{text}</b><br>%{x:.4f} → %{y:.3f}<extra></extra>",
    ))

    if model and model.get("predict_fn") is not None and len(df_clean) > 1:
        x_range = np.linspace(df_clean[x_col].min(), df_clean[x_col].max(), 300)
        y_pred  = model["predict_fn"](x_range)
        r2      = model.get("r_squared", 0)
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_pred,
            mode="lines",
            name=f"{model.get('model_type', '')} (R²={r2:.4f})",
            line=dict(color="#EF5350", width=2.5),
        ))
        fig.add_annotation(
            text=(f"<b>{model.get('formula_str', '')}</b><br>"
                  f"R² = {r2:.4f} | RMSE = {model.get('rmse', 'N/A')}"),
            xref="paper", yref="paper", x=0.03, y=0.97,
            showarrow=False, align="left",
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#BDBDBD", borderwidth=1,
        )

    fig.update_layout(
        title=f"{index_name} vs {y_col.upper()}",
        xaxis_title=f"Índice {index_name} (satelital)",
        yaxis_title=f"{y_col} (campo)",
        template="plotly_white",
        height=440,
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig


def plot_regression_comparison(comparison_df: pd.DataFrame, y_col: str = "parámetro") -> go.Figure:
    """
    Gráfico de barras comparando R² de los 4 modelos de regresión
    para un par de variables.
    """
    if comparison_df.empty:
        return go.Figure()

    colors = ["#1565C0", "#2196F3", "#FF9800", "#EF5350"]
    fig = go.Figure()
    for i, row in comparison_df.iterrows():
        r2 = row["R²"] if row["R²"] is not None else 0
        fig.add_trace(go.Bar(
            x=[row["tipo_regresión"]],
            y=[r2],
            name=row["tipo_regresión"],
            marker_color=colors[i % 4],
            text=[f"R²={r2:.4f}"],
            textposition="outside",
        ))

    fig.update_layout(
        title=f"Comparación de modelos – {y_col}",
        yaxis_title="R²",
        yaxis_range=[0, 1.1],
        template="plotly_white",
        showlegend=False,
        height=340,
    )
    return fig


def plot_pairwise_scatter(
    merged_df: pd.DataFrame,
    sat_col: str,
    field_cols: list[str],
    models: Optional[dict] = None,
) -> go.Figure:
    """
    Cuadrícula de dispersión: un panel por cada parámetro de campo
    vs el índice satelital seleccionado.

    Parámetros
    ----------
    models : dict {field_col: model_dict} con modelos preajustados (opcional)
    """
    n = len(field_cols)
    if n == 0:
        return go.Figure()

    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=field_cols,
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
    )

    for idx, f_col in enumerate(field_cols):
        row = idx // cols + 1
        col = idx %  cols + 1
        df_c = merged_df[[sat_col, f_col]].dropna()
        if df_c.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=df_c[sat_col], y=df_c[f_col],
                mode="markers",
                marker=dict(size=8, color="#1565C0", opacity=0.8),
                showlegend=False,
                name=f_col,
            ),
            row=row, col=col,
        )

        if models and f_col in models and models[f_col].get("predict_fn"):
            m = models[f_col]
            x_r = np.linspace(df_c[sat_col].min(), df_c[sat_col].max(), 200)
            y_r = m["predict_fn"](x_r)
            fig.add_trace(
                go.Scatter(
                    x=x_r, y=y_r,
                    mode="lines",
                    line=dict(color="#EF5350", width=2),
                    showlegend=False,
                    name=f"modelo {f_col}",
                ),
                row=row, col=col,
            )
            # Anotación R²
            fig.add_annotation(
                text=f"R²={m['r_squared']:.3f}",
                xref=f"x{idx+1 if idx > 0 else ''}",
                yref=f"y{idx+1 if idx > 0 else ''}",
                x=df_c[sat_col].max(),
                y=df_c[f_col].min(),
                showarrow=False,
                font=dict(size=10, color="#EF5350"),
                row=row, col=col,
            )

        fig.update_xaxes(title_text=sat_col, row=row, col=col)
        fig.update_yaxes(title_text=f_col,   row=row, col=col)

    fig.update_layout(
        title=f"Parámetros de campo vs {sat_col}",
        template="plotly_white",
        height=320 * rows,
    )
    return fig


def plot_residuals(
    x: np.ndarray,
    y: np.ndarray,
    model: dict,
    x_label: str = "índice_sat",
    y_label: str = "parámetro",
) -> go.Figure:
    """Gráfico de residuos para diagnóstico del modelo."""
    fn = model.get("predict_fn")
    if fn is None:
        return go.Figure()

    mask = ~(np.isnan(x) | np.isnan(y))
    x_c, y_c = x[mask], y[mask]
    y_hat = fn(x_c)
    residuals = y_c - y_hat

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Residuos vs Ajustados", "Q-Q Plot de residuos"])

    fig.add_trace(
        go.Scatter(x=y_hat, y=residuals, mode="markers",
                   marker=dict(color="#1565C0", size=8, opacity=0.8),
                   name="Residuos"),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # Q-Q plot
    (osm, osr), (slope, intercept, _) = sp_stats.probplot(residuals)
    fig.add_trace(
        go.Scatter(x=osm, y=osr, mode="markers",
                   marker=dict(color="#1565C0", size=6),
                   name="Cuantiles"),
        row=1, col=2,
    )
    qq_line = np.array([min(osm), max(osm)]) * slope + intercept
    fig.add_trace(
        go.Scatter(x=[min(osm), max(osm)], y=qq_line,
                   mode="lines", line=dict(color="red", dash="dash"),
                   name="Línea teórica"),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Valores ajustados", row=1, col=1)
    fig.update_yaxes(title_text="Residuos",          row=1, col=1)
    fig.update_xaxes(title_text="Cuantiles teóricos", row=1, col=2)
    fig.update_yaxes(title_text="Cuantiles muestra",  row=1, col=2)
    fig.update_layout(
        title=f"Diagnóstico de residuos – modelo {model.get('model_type', '')}",
        template="plotly_white",
        height=360,
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def export_model(model: dict, path: str = "outputs/modelo_correlacion.pkl") -> str:
    """Exporta los coeficientes del modelo (sin lambdas) con joblib."""
    import joblib
    exportable = {k: v for k, v in model.items() if k != "predict_fn"}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(exportable, path)
    logger.info("Modelo exportado: %s", path)
    return path
