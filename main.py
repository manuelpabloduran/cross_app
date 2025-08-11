# Version: 2.2
# Doble selector (Inicio/Final), botón para mostrar/ocultar selector,
# filtro combinado/único y head del DF filtrado + funnel desde charts_cross.py

import os, base64
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from charts_cross import funnel_por_tipo  # <- módulo externo con funciones de gráficos

st.set_page_config(page_title="Zonas Inicio/Final + Filtro + Gráficas", page_icon="⚽", layout="wide")

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
if "zone_inicio" not in st.session_state:
    st.session_state.zone_inicio = None
if "zone_final" not in st.session_state:
    st.session_state.zone_final = None
if "show_selector" not in st.session_state:
    st.session_state.show_selector = False

# ---------- figura base ----------
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
    # malla de puntos "fantasma" para que el box-select emita eventos
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

    # lista de puntos o dict con 'points'
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
    return {
        "x0": round(clamp(x0), 2), "x1": round(clamp(x1), 2),
        "y0": round(clamp(y0), 2), "y1": round(clamp(y1), 2),
    }

# ---------- carga del CSV ----------
def load_cross_stats():
    candidates = ["cross_stats.csv", "data/cross_stats.csv", "dataset/cross_stats.csv"]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        st.error("No encontré 'cross_stats.csv' (probé en raíz, /data y /dataset).")
        return None
    df = pd.read_csv(path)
    # asegurar tipos numéricos en posiciones
    for c in ["x", "y", "endX", "endY"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def apply_filters(df: pd.DataFrame, zi: dict | None, zf: dict | None):
    """Combinado si hay ambos; si hay uno, sólo ese; si ninguno, devuelve df."""
    if df is None:
        return None
    filt = df
    if zi:
        filt = filt[filt["x"].between(zi["x0"], zi["x1"]) & filt["y"].between(zi["y0"], zi["y1"])]
    if zf:
        filt = filt[filt["endX"].between(zf["x0"], zf["x1"]) & filt["endY"].between(zf["y0"], zf["y1"])]
    return filt

# ---------- UI ----------
st.title("Zonas de INICIO y FINALIZACIÓN + filtro combinado + gráficas")

# Botón para mostrar/ocultar selector
btn_label = "Seleccionar Zona Análisis" if not st.session_state.show_selector else "Ocultar Selector de Zonas"
if st.button(btn_label):
    st.session_state.show_selector = not st.session_state.show_selector

# Selector (sólo visible si está activo)
if st.session_state.show_selector:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("### Inicio de centro")
        fig_i = build_fig(st.session_state.zone_inicio)
        zone_i = capture_box(fig_i, "pitch_inicio")
        if zone_i:
            st.session_state.zone_inicio = zone_i
        z = st.session_state.zone_inicio
        st.write("Zona inicio:", z if z else "—")
        if st.button("🧹 Limpiar inicio"):
            st.session_state.zone_inicio = None
            st.rerun()
    with c2:
        st.markdown("### Finalización de centro")
        fig_f = build_fig(st.session_state.zone_final)
        zone_f = capture_box(fig_f, "pitch_final")
        if zone_f:
            st.session_state.zone_final = zone_f
        z = st.session_state.zone_final
        st.write("Zona finalización:", z if z else "—")
        if st.button("🧹 Limpiar finalización"):
            st.session_state.zone_final = None
            st.rerun()

st.divider()

# Cargar DF y aplicar filtro
df = load_cross_stats()
if df is not None:
    zi, zf = st.session_state.zone_inicio, st.session_state.zone_final
    df_scope = apply_filters(df, zi, zf)

    st.subheader("Validación del filtro (head)")
    st.caption(f"Filas resultantes: {len(df_scope)} de {len(df)}")
    st.dataframe(df_scope.head(), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Gráficas (sobre el DF filtrado si hay filtro, o sobre el total)")

    # Funnel Abierto vs Cerrado
    try:
        fig_funnel = funnel_por_tipo(df_scope, open_label="Abierto", closed_label="Cerrado")
        st.plotly_chart(fig_funnel, use_container_width=True)
    except Exception as e:
        st.warning(f"No se pudo construir el funnel: {e}")