# Version: 2.3
# Filtros generales (panel izq) + doble selector opcional + filtro combinado/único
# y gráficas sobre df_scope. Requiere charts_cross.py (funnel_por_tipo).

import os, base64
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from charts_cross import funnel_por_tipo  # funciones de gráficos

st.set_page_config(page_title="Análisis de Centros • Filtros + Zonas", page_icon="⚽", layout="wide")

# ---------- cargar imagen 'pitch_opta' ----------
CANDIDATES_IMG = ["pitch_opta.png", "pitch_opta.jpg", "pitch_opta.jpeg"]
IMG_PATH = next((p for p in CANDIDATES_IMG if os.path.exists(p)), None)
if not IMG_PATH:
    st.error("No encontré 'pitch_opta.(png/jpg/jpeg)' en la raíz del repo.")
    st.stop()

def image_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = path.split(".")[-1].lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
    return f"data:image/{mime};base64,{b64}"

# ---------- estado ----------
ss = st.session_state
ss.setdefault("zone_inicio", None)
ss.setdefault("zone_final", None)
ss.setdefault("show_selector", False)

# ---------- carga del CSV ----------
def load_cross_stats():
    candidates = ["cross_stats.csv", "data/cross_stats.csv", "dataset/cross_stats.csv"]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        st.error("No encontré 'cross_stats.csv' (probé en raíz, /data y /dataset).")
        return None
    df = pd.read_csv(path)
    # asegurar tipos numéricos
    for c in ["x", "y", "endX", "endY", "xg_corrected"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------- figura base (pitch con box-select) ----------
def build_fig(zone=None):
    fig = go.Figure()
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0,100], visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[0,100], visible=False),   # (0,0) abajo-izq; Y↑
        dragmode="select"
    )
    fig.add_layout_image(dict(
        source=image_data_url(IMG_PATH),
        xref="x", yref="y",
        x=0, y=0, sizex=100, sizey=100,
        xanchor="left", yanchor="bottom",
        sizing="stretch",
        layer="below"
    ))
    # malla de puntos “fantasma” para que el box-select emita eventos
    step = 1
    xs = [xx for xx in range(0,101,step) for _ in range(0,101,step)]
    ys = [yy for _ in range(0,101,step) for yy in range(0,101,step)]
    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode="markers",
        marker=dict(size=5, opacity=0.01),
        hoverinfo="skip", showlegend=False
    ))
    # dibujar zona si existe
    if zone:
        fig.add_shape(
            type="rect",
            x0=zone["x0"], x1=zone["x1"], y0=zone["y0"], y1=zone["y1"],
            line=dict(color="orange", width=2),
            fillcolor="rgba(255,165,0,0.30)",
            layer="above",
        )
    return fig

def capture_box(fig, widget_key: str):
    """Devuelve dict con x0,x1,y0,y1 (o None) usando box select."""
    try:
        from streamlit_plotly_events2 import plotly_events
    except Exception:
        from streamlit_plotly_events import plotly_events

    selected = plotly_events(
        fig,
        select_event=True, click_event=False, hover_event=False,
        override_width="100%", override_height=520,
        key=widget_key
    )
    if not selected:
        return None
    if isinstance(selected, list) and selected and isinstance(selected[0], dict) and "x" in selected[0]:
        pts = selected
    elif isinstance(selected[-1], dict) and "points" in selected[-1]:
        pts = selected[-1]["points"]
    else:
        pts = []
    if not pts:
        return None
    xs = [float(p["x"]) for p in pts]
    ys = [float(p["y"]) for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    clamp = lambda v: max(0.0, min(100.0, v))
    return {"x0": round(clamp(x0), 2), "x1": round(clamp(x1), 2),
            "y0": round(clamp(y0), 2), "y1": round(clamp(y1), 2)}

# ---------- filtros generales (panel izquierdo) ----------
FILTER_KEYS = {
    "team": "TeamName",
    "rival": "TeamRival",
    "temporada": "Temporada",
    "competencia": "Competencia",
    "finalizacion": "ultimo_event_name",
    "xg": "xg_corrected",
}

def general_filter_panel(df: pd.DataFrame):
    st.markdown("### Filtros generales")
    selected = {}
    # Helper para multiselect seguro
    def ms(label, colname, key):
        if colname in df.columns:
            opts = sorted([x for x in df[colname].dropna().unique().tolist()])
            val = st.multiselect(label, options=opts, default=[], key=key)
            selected[key] = (colname, val)
        else:
            st.caption(f"— columna no encontrada: **{colname}**")
            selected[key] = (None, [])
    # Multiselects
    ms("Equipo", FILTER_KEYS["team"], "flt_team")
    ms("Rival", FILTER_KEYS["rival"], "flt_rival")
    ms("Temporada", FILTER_KEYS["temporada"], "flt_temporada")
    ms("Competencia", FILTER_KEYS["competencia"], "flt_competencia")
    ms("Finalización (ultimo_event_name)", FILTER_KEYS["finalizacion"], "flt_final")

    # Slider xG
    xg_col = FILTER_KEYS["xg"]
    if xg_col in df.columns:
        xg_series = df[xg_col].dropna()
        if len(xg_series):
            xg_min = float(xg_series.min())
            xg_max = float(xg_series.max())
            # inicializar default del slider
            ss.setdefault("flt_xg_default", (xg_min, xg_max))
            rng = st.slider(
                "Rango xG (xg_corrected)",
                min_value=xg_min, max_value=xg_max,
                value=ss.get("flt_xg", ss["flt_xg_default"]),
                step=0.01, key="flt_xg"
            )
            selected["flt_xg"] = (xg_col, rng)
        else:
            st.caption("— xg_corrected sin datos numéricos")
            selected["flt_xg"] = (None, None)
    else:
        st.caption("— columna no encontrada: **xg_corrected**")
        selected["flt_xg"] = (None, None)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Limpiar filtros"):
            for k in ["flt_team","flt_rival","flt_temporada","flt_competencia","flt_final"]:
                ss[k] = []
            if "flt_xg_default" in ss:
                ss["flt_xg"] = ss["flt_xg_default"]
            st.rerun()
    with col_b:
        btn_label = "Seleccionar Zona Análisis" if not ss.show_selector else "Ocultar Selector de Zonas"
        if st.button(btn_label):
            ss.show_selector = not ss.show_selector
            st.rerun()

    return selected

def apply_general_filters(df: pd.DataFrame, sel: dict):
    if df is None:
        return None
    out = df.copy()
    # multiselect inclusivos
    for k in ["flt_team","flt_rival","flt_temporada","flt_competencia","flt_final"]:
        colname, values = sel.get(k, (None, []))
        if colname and values:
            out = out[out[colname].isin(values)]
    # rango xG
    colname, rng = sel.get("flt_xg", (None, None))
    if colname and isinstance(rng, tuple):
        lo, hi = float(rng[0]), float(rng[1])
        out = out[out[colname].between(lo, hi)]
    return out

# ---------- filtros por zonas ----------
def apply_zone_filters(df: pd.DataFrame, zi: dict | None, zf: dict | None):
    """Combinado si hay ambos; si hay uno, sólo ese; si ninguno, devuelve df."""
    if df is None:
        return None
    filt = df
    if zi:
        if not {"x","y"}.issubset(filt.columns):
            st.warning("Falta 'x' o 'y' para filtrar por INICIO.")
        else:
            filt = filt[filt["x"].between(zi["x0"], zi["x1"]) & filt["y"].between(zi["y0"], zi["y1"])]
    if zf:
        if not {"endX","endY"}.issubset(filt.columns):
            st.warning("Falta 'endX' o 'endY' para filtrar por FINALIZACIÓN.")
        else:
            filt = filt[filt["endX"].between(zf["x0"], zf["x1"]) & filt["endY"].between(zf["y0"], zf["y1"])]
    return filt

# ===================== UI =====================

st.title("Análisis de Centros")

df = load_cross_stats()
if df is None:
    st.stop()

# Layout: panel izquierdo (filtros) / derecho (selector + resultados)
col_filters, col_main = st.columns([0.28, 0.72], gap="large")

with col_filters:
    selections = general_filter_panel(df)

with col_main:
    # Selector de zonas (opcional)
    if ss.show_selector:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("#### Inicio de centro")
            fig_i = build_fig(ss.zone_inicio)
            zone_i = capture_box(fig_i, "pitch_inicio")
            if zone_i:
                ss.zone_inicio = zone_i
            st.caption(f"Zona inicio: {ss.zone_inicio if ss.zone_inicio else '—'}")
            if st.button("🧹 Limpiar inicio"):
                ss.zone_inicio = None
                st.rerun()
        with c2:
            st.markdown("#### Finalización de centro")
            fig_f = build_fig(ss.zone_final)
            zone_f = capture_box(fig_f, "pitch_final")
            if zone_f:
                ss.zone_final = zone_f
            st.caption(f"Zona finalización: {ss.zone_final if ss.zone_final else '—'}")
            if st.button("🧹 Limpiar finalización"):
                ss.zone_final = None
                st.rerun()

    st.divider()

    # Aplicar filtros: primero generales, luego zonas
    df_after_general = apply_general_filters(df, selections)
    df_scope = apply_zone_filters(df_after_general, ss.zone_inicio, ss.zone_final)

    st.subheader("Situaciones principales")
    st.caption(f"Filas resultantes: {len(df_scope)} de {len(df)}")
    st.dataframe(df_scope.sort_values(by='xg_corrected', ascending=False), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Gráficas")

    try:
        fig_funnel = funnel_por_tipo(df_scope, open_label="Abierto", closed_label="Cerrado")
        st.plotly_chart(fig_funnel, use_container_width=True)
    except Exception as e:
        st.warning(f"No se pudo construir el funnel: {e}")