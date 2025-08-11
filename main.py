# main.py
import base64, os
import streamlit as st
import plotly.graph_objects as go

# --- config ---
st.set_page_config(page_title="Rectángulo en pitch Opta", page_icon="⚽", layout="centered")

# --- ruta de tu imagen ---
# Colocá 'pitch_opta.png' (o .jpg) en la carpeta raíz del repo/app.
CANDIDATES = ["pitch_opta.png", "pitch_opta.jpg", "pitch_opta.jpeg"]
PITCH_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
if not PITCH_PATH:
    st.error("No encontré 'pitch_opta' (png/jpg) en el repo. Ponelo en la raíz y recargá.")
    st.stop()

# --- util: leer imagen y pasarla a data URL ---
def image_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = path.split(".")[-1].lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
    return f"data:image/{mime};base64,{b64}"

# --- figura: fondo + capa de puntos invisibles para selección ---
def build_fig(zone=None):
    fig = go.Figure()
    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0,100], constrain="domain", scaleanchor="y", scaleratio=1, visible=False),
        yaxis=dict(range=[100,0], visible=False),  # Opta: (0,0) arriba-izq
        dragmode="select"  # arrastrar = box select
    )
    fig.add_layout_image(dict(
        source=image_data_url(PITCH_PATH),
        xref="x", yref="y",
        x=0, y=100, sizex=100, sizey=100,
        xanchor="left", yanchor="top",
        sizing="stretch",
        layer="below"
    ))
    # malla de puntos "invisible" (seleccionables) para que funcione box select
    step = 1  # precisión 1 unidad Opta → 101x101 = 10201 puntos (ok con Scattergl)
    xs = [xx for xx in range(0,101,step) for _ in range(0,101,step)]
    ys = [yy for _ in range(0,101,step) for yy in range(0,101,step)]
    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode="markers",
        marker=dict(size=5, opacity=0.01),  # casi invisibles pero seleccionables
        hoverinfo="skip", showlegend=False
    ))
    # si ya hay zona guardada, dibujarla (opcional)
    if zone:
        fig.add_shape(
            type="rect",
            x0=zone["x0"], x1=zone["x1"], y0=zone["y0"], y1=zone["y1"],
            line=dict(color="orange", width=2),
            fillcolor="rgba(255,165,0,0.30)",
            layer="above",
        )
    return fig

# --- estado ---
if "zone" not in st.session_state:
    st.session_state.zone = None

st.title("Seleccioná un rectángulo en tu pitch Opta (0–100)")
st.caption("Arrastrá dentro del campo para box-select. El eje Y está invertido: 0 arriba → 100 abajo.")

fig = build_fig(zone=st.session_state.zone)

# --- capturar selección ---
try:
    from streamlit_plotly_events2 import plotly_events  # si lo tenés
except Exception:
    from streamlit_plotly_events import plotly_events   # lib estándar

selected = plotly_events(
    fig,
    select_event=True, click_event=False, hover_event=False,
    override_width="100%", override_height=620, key="pitch"
)

# --- calcular bounding box y mostrar ---
if selected:
    # Soportar ambas formas de retorno (lista de puntos o dict con 'points')
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
        st.session_state.zone = {
            "x0": round(clamp(x0), 2),
            "x1": round(clamp(x1), 2),
            "y0": round(clamp(y0), 2),
            "y1": round(clamp(y1), 2),
        }

# --- output mínimo ---
if st.session_state.zone:
    z = st.session_state.zone
    st.success(f"Zona → x0={z['x0']}, x1={z['x1']} • y0={z['y0']}, y1={z['y1']}")
else:
    st.info("Dibujá un rectángulo para ver las coordenadas acá.")
