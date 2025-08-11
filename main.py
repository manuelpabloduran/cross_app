# Version: 2
# Doble selector de zona (Inicio / Finalización) sobre 'pitch_opta'
# Sistema de ejes: (0,0) abajo-izq; X→derecha; Y→arriba

import os, base64
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Selección zonas: Inicio/Finalización", page_icon="⚽", layout="wide")

# ---------- cargar imagen 'pitch_opta' ----------
CANDIDATES = ["pitch_opta.png", "pitch_opta.jpg", "pitch_opta.jpeg"]
IMG_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
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

# ---------- figura base ----------
def build_fig(zone=None):
    fig = go.Figure()
    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0,100], visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[0,100], visible=False),   # Y crece hacia arriba
        dragmode="select"                           # arrastrar = box select
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
    step = 1  # precisión: 1 unidad Opta
    xs = [xx for xx in range(0,101,step) for _ in range(0,101,step)]
    ys = [yy for _ in range(0,101,step) for yy in range(0,101,step)]
    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode="markers",
        marker=dict(size=5, opacity=0.01),  # casi invisibles pero seleccionables
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

# ---------- captura de selección (box select) ----------
def select_zone_ui(title: str, state_key: str, widget_key: str):
    st.markdown(f"### {title}")
    fig = build_fig(st.session_state.get(state_key))
    try:
        from streamlit_plotly_events2 import plotly_events
    except Exception:
        from streamlit_plotly_events import plotly_events

    selected = plotly_events(
        fig,
        select_event=True, click_event=False, hover_event=False,
        override_width="100%", override_height=620,
        key=widget_key
    )

    # parsear selección -> bounding box
    if selected:
        if isinstance(selected, list) and selected and isinstance(selected[0], dict) and "x" in selected[0]:
            pts = selected
        elif isinstance(selected[-1], dict) and "points" in selected[-1]:
            pts = selected[-1]["points"]
        else:
            pts = []

        if pts:
            xs = [float(p["x"]) for p in pts]
            ys = [float(p["y"]) for p in pts]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            clamp = lambda v: max(0.0, min(100.0, v))
            st.session_state[state_key] = {
                "x0": round(clamp(x0), 2), "x1": round(clamp(x1), 2),
                "y0": round(clamp(y0), 2), "y1": round(clamp(y1), 2),
            }

    # outputs y acciones
    z = st.session_state.get(state_key)
    col1, col2 = st.columns([1.4, 1])
    with col1:
        if z:
            st.success(f"{title}: x0={z['x0']}  x1={z['x1']}  •  y0={z['y0']}  y1={z['y1']}")
        else:
            st.info("Arrastrá un rectángulo para capturar la zona.")
    with col2:
        if st.button(f"🧹 Limpiar {title}", key=f"clear_{state_key}"):
            st.session_state[state_key] = None
            st.rerun()

# ---------- UI ----------
st.title("Zonas de INICIO y FINALIZACIÓN de centro")
st.caption("Sistema de ejes: (0,0) abajo-izq • X→derecha • Y→arriba")

c1, c2 = st.columns(2, gap="large")
with c1:
    select_zone_ui("Inicio de centro", "zone_inicio", "pitch_inicio")
with c2:
    select_zone_ui("Finalización de centro", "zone_final", "pitch_final")

st.divider()
st.subheader("Resumen")
z1, z2 = st.session_state.zone_inicio, st.session_state.zone_final
st.write({
    "inicio": z1 if z1 else None,
    "finalizacion": z2 if z2 else None
})