# main.py
import io, base64
import streamlit as st
import plotly.graph_objects as go
from mplsoccer import Pitch

# plotly-events: v2 -> fallback a v1
try:
    from streamlit_plotly_events2 import plotly_events
except Exception:
    from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Rectángulo en pitch Opta", page_icon="⚽", layout="centered")

# --- helper: imagen del pitch Opta (100x100) como base64 data URL ---
def pitch_data_url():
    pitch = Pitch(pitch_type="opta", pitch_length=100, pitch_width=100, line_zorder=2)
    fig, _ = pitch.draw(figsize=(7.2, 5.0), tight_layout=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return "data:image/png;base64," + b64

# --- build figure: fondo + capa invisible de puntos para habilitar box select ---
def build_figure(zone=None):
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(range=[0, 100], constrain="domain", scaleanchor="y", scaleratio=1, visible=False),
        yaxis=dict(range=[100, 0], visible=False),   # Opta: origen arriba-izquierda
        margin=dict(l=10, r=10, t=10, b=10),
        dragmode="select"  # arrastrar para dibujar rectángulo de selección
    )
    # fondo del campo
    fig.add_layout_image(dict(
        source=pitch_data_url(), xref="x", yref="y",
        x=0, y=100, sizex=100, sizey=100, sizing="stretch", layer="below"
    ))
    # capa de puntos invisibles (permite que plotly emita 'selectedData' con el box)
    step = 2  # 51x51=2601 puntos (liviano)
    xs, ys = [], []
    for xx in range(0, 101, step):
        for yy in range(0, 101, step):
            xs.append(xx); ys.append(yy)
    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode="markers",
        marker=dict(size=1, opacity=0),
        hoverinfo="skip", showlegend=False
    ))
    # si hay zona previa, la dibujamos para validación visual
    if zone:
        fig.add_shape(
            type="rect",
            x0=zone["x0"], x1=zone["x1"], y0=zone["y0"], y1=zone["y1"],
            line=dict(color="orange", width=2),
            fillcolor="rgba(255,165,0,0.3)",
            layer="above",
        )
    return fig

# estado: una única zona
if "zone" not in st.session_state:
    st.session_state.zone = None

st.title("Seleccionar rectángulo en pitch Opta 100×100")
st.caption("Arrastrá con el mouse para dibujar el rectángulo. Eje Y invertido (0 arriba → 100 abajo).")

# figura
fig = build_figure(zone=st.session_state.zone)

# eventos: usamos 'select_event' para capturar el box select
ev = plotly_events(
    fig,
    select_event=True, click_event=False, hover_event=False,
    override_width="100%", key="pitch"
)

# procesamos el último evento (si hay)
if ev:
    last = ev[-1]
    x0 = y0 = x1 = y1 = None
    # 1) Si viene 'range' (lo más directo)
    if isinstance(last, dict) and "range" in last and last["range"]:
        rx = last["range"].get("x")
        ry = last["range"].get("y")
        if rx and ry:
            x0, x1 = float(min(rx)), float(max(rx))
            y0, y1 = float(min(ry)), float(max(ry))
    # 2) Fallback: calcular por puntos seleccionados
    if (x0 is None or y0 is None) and "points" in last and last["points"]:
        xs = [float(p["x"]) for p in last["points"]]
        ys = [float(p["y"]) for p in last["points"]]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)

    # guardamos si tenemos límites válidos
    if None not in (x0, x1, y0, y1):
        # recortamos a 0..100 y redondeamos
        clamp = lambda v: max(0.0, min(100.0, v))
        z = {
            "x0": round(clamp(x0), 2),
            "x1": round(clamp(x1), 2),
            "y0": round(clamp(y0), 2),
            "y1": round(clamp(y1), 2),
        }
        st.session_state.zone = z

# salida mínima: printear coordenadas guardadas
if st.session_state.zone:
    z = st.session_state.zone
    st.success(f"Zona: x0={z['x0']}, x1={z['x1']} • y0={z['y0']}, y1={z['y1']}")
else:
    st.info("Dibujá un rectángulo para ver las coordenadas aquí.")
