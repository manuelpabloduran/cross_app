# main.py
import os, base64
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Box select en pitch Opta (0,0 abajo-izq)", page_icon="⚽")

# --------- Cargar tu imagen ---------
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

# --------- Figura base (sin necesidad de pan) ---------
def build_base_fig():
    fig = go.Figure()
    # Ejes: 0..100 en X y 0..100 en Y (Y hacia arriba). ¡Nada invertido!
    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, 100], visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[0, 100], visible=False),
        dragmode="select"   # arrastrar = box select
    )
    # Imagen anclada a ESQUINA INFERIOR IZQUIERDA (xanchor=left, yanchor=bottom)
    fig.add_layout_image(dict(
        source=image_data_url(IMG_PATH),
        xref="x", yref="y",
        x=0, y=0,                  # esquina inferior-izquierda
        sizex=100, sizey=100,      # cubre 0..100 x 0..100
        xanchor="left", yanchor="bottom",
        sizing="stretch",
        layer="below"
    ))
    # Malla de puntos “fantasma” (seleccionables) para que el box select emita eventos
    step = 1  # precisión de 1 unidad Opta
    xs = [xx for xx in range(0, 101, step) for _ in range(0, 101, step)]
    ys = [yy for _ in range(0, 101, step) for yy in range(0, 101, step)]
    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode="markers",
        marker=dict(size=5, opacity=0.01),  # casi invisibles, pero seleccionables
        hoverinfo="skip", showlegend=False
    ))
    return fig

# Estado: la última zona seleccionada
if "zone" not in st.session_state:
    st.session_state.zone = None

st.title("Seleccioná un rectángulo (0,0 abajo-izq • X→derecha • Y→arriba)")
st.caption("Arrastrá dentro del campo para box-select. No hace falta panear.")

fig = build_base_fig()

# Capturar selección
try:
    from streamlit_plotly_events2 import plotly_events
except Exception:
    from streamlit_plotly_events import plotly_events

selected = plotly_events(
    fig,
    select_event=True, click_event=False, hover_event=False,
    override_width="100%", override_height=620, key="pitch"
)

# Calcular bounding box (x0,x1,y0,y1) y guardarlo
if selected:
    # Algunas versiones devuelven lista de puntos; otras, dict con 'points'
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
            "x0": round(clamp(x0), 2), "x1": round(clamp(x1), 2),
            "y0": round(clamp(y0), 2), "y1": round(clamp(y1), 2),
        }

# Output mínimo
if st.session_state.zone:
    z = st.session_state.zone
    st.success(f"Zona → x0={z['x0']}, x1={z['x1']} • y0={z['y0']}, y1={z['y1']}")
else:
    st.info("Dibujá un rectángulo para ver las coordenadas acá.")