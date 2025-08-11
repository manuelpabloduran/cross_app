import io, base64
import streamlit as st
import plotly.graph_objects as go
from mplsoccer import Pitch
from streamlit_plotly_events import plotly_events

# --- helper: pitch Opta sin márgenes ---
import io, base64
from mplsoccer import Pitch

def pitch_data_url():
    pitch = Pitch(pitch_type="opta", pitch_length=100, pitch_width=100, line_zorder=2)
    fig, ax = pitch.draw(figsize=(7.2, 5.0))
    # ocupar todo el lienzo (sin bordes de Matplotlib)
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_alpha(0)  # fondo transparente
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

# --- figura Plotly con alto fijo y anclas explícitas ---
import plotly.graph_objects as go

def build_figure():
    fig = go.Figure()
    fig.update_layout(
        height=620,  # <- importante para evitar “scroll raro”
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, 100], constrain="domain", scaleanchor="y", scaleratio=1, visible=False),
        yaxis=dict(range=[100, 0], visible=False),
        dragmode="select"
    )
    fig.add_layout_image(dict(
        source=pitch_data_url(),
        xref="x", yref="y",
        x=0, y=100, sizex=100, sizey=100,
        xanchor="left", yanchor="top",   # <- anclas explícitas
        sizing="stretch", layer="below"
    ))
    # capa de puntos invisibles para que funcione el box select
    step = 2
    xs, ys = [], []
    for xx in range(0, 101, step):
        for yy in range(0, 101, step):
            xs.append(xx); ys.append(yy)
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=3, opacity=0),
        hoverinfo="skip", showlegend=False
    ))
    return fig


# 3) estado (la última zona)
if "zone" not in st.session_state:
    st.session_state.zone = None

st.set_page_config(page_title="Box select en pitch Opta", page_icon="⚽")
st.title("Seleccionar un rectángulo en un pitch Opta (100×100)")
st.caption("Arrastrá un rectángulo en el área del campo. Eje Y invertido: 0 arriba → 100 abajo.")

fig = build_figure()
selected = plotly_events(
    fig,
    select_event=True, click_event=False, hover_event=False,
    override_width="100%", override_height=620,  # <- fija altura también aquí
    key="pitch"
)


# 5) calcular bounding box y mostrar
if selected:
    # Algunas versiones devuelven directamente una lista de puntos [{'x':..,'y':..}, ...]
    # Otras envuelven en {'points': [...]} -> soportamos ambas.
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
        z = {"x0": round(clamp(x0), 2), "x1": round(clamp(x1), 2),
             "y0": round(clamp(y0), 2), "y1": round(clamp(y1), 2)}
        st.session_state.zone = z

if st.session_state.zone:
    z = st.session_state.zone
    st.success(f"Zona seleccionada → x0={z['x0']}, x1={z['x1']} • y0={z['y0']}, y1={z['y1']}")
else:
    st.info("Dibujá un rectángulo para ver las coordenadas acá.")
