# main.py
import io
import json
import base64
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from mplsoccer import Pitch

# Intentamos importar el componente de eventos (v2 o el original)
try:
    from streamlit_plotly_events2 import plotly_events
except Exception:
    from streamlit_plotly_events import plotly_events

# ---------------- Config ---------------- #
st.set_page_config(page_title="Análisis de centros • Selección Opta 100x100", page_icon="⚽", layout="wide")

# Estado
if "clicks" not in st.session_state:
    st.session_state.clicks = []  # cada item: {"x": float, "y": float, "ts": str}

# ---------------- Sidebar ---------------- #
st.sidebar.header("Parámetros del campo")
grid_on = st.sidebar.toggle("Mostrar grilla guía", value=True)
grid_step = st.sidebar.slider("Paso de grilla (unidad Opta)", 2, 25, 10, step=1)
snap = st.sidebar.toggle("Snap a grilla (redondear al paso)", value=False)
accumulate = st.sidebar.toggle("Acumular clics", value=True)

if st.sidebar.button("🧹 Limpiar puntos"):
    st.session_state.clicks = []
    st.sidebar.success("Puntos borrados")

# ---------------- Utilidades ---------------- #
def mpl_pitch_png_base64(pitch_len=100, pitch_wid=100, figsize=(7.2, 5.0), dpi=160):
    """Dibuja un pitch Opta con mplsoccer y devuelve un data URL base64 PNG."""
    pitch = Pitch(pitch_type="opta", pitch_length=pitch_len, pitch_width=pitch_wid, line_zorder=2)
    fig_mpl, _ = pitch.draw(figsize=figsize, tight_layout=False)
    buf = io.BytesIO()
    fig_mpl.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig_mpl.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{b64}"

def build_pitch_figure(show_grid=True, step=10, clicks=None):
    """Crea figura Plotly 0..100 x 0..100 (Y invertida) con imagen de fondo + grilla opcional + puntos guardados."""
    # 1) Base
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(range=[0, 100], constrain="domain", scaleanchor="y", scaleratio=1, visible=False),
        yaxis=dict(range=[100, 0], visible=False),  # Opta: origen arriba-izquierda
        margin=dict(l=10, r=10, t=10, b=10),
        dragmode="pan"
    )

    # 2) Fondo: imagen del pitch (mplsoccer)
    src = mpl_pitch_png_base64()
    fig.add_layout_image(
        dict(
            source=src, xref="x", yref="y",
            x=0, y=100, sizex=100, sizey=100,
            sizing="stretch", layer="below"
        )
    )

    # 3) Grilla guía
    shapes = []
    if show_grid and step > 0:
        ticks = list(range(0, 101, step))
        # Verticales
        for x in ticks:
            shapes.append(dict(type="line", x0=x, y0=0, x1=x, y1=100, line=dict(width=1, dash="dot")))
        # Horizontales
        for y in ticks:
            shapes.append(dict(type="line", x0=0, y0=y, x1=100, y1=y, line=dict(width=1, dash="dot")))
    if shapes:
        fig.update_layout(shapes=shapes)

    # 4) Puntos ya guardados
    if clicks and len(clicks) > 0:
        xs = [p["x"] for p in clicks]
        ys = [p["y"] for p in clicks]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text", text=[str(i+1) for i in range(len(xs))],
            textposition="top center", marker=dict(size=10),
            name="Seleccionados", hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>"
        ))
    else:
        # capa vacía para habilitar eventos de click
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False))

    return fig

def snap_value(v, step):
    """Redondea v al múltiplo más cercano de 'step' entre 0 y 100."""
    if step <= 0:
        return v
    return max(0.0, min(100.0, round(v / step) * step))

# ---------------- Tabs ---------------- #
tab_general, tab_equipos, tab_jugadores = st.tabs([
    "Análisis general de rendimiento",
    "Rendimiento Equipos",
    "Rendimiento Jugadores"
])

# ---------------- Tab 1: Selección 100x100 ---------------- #
with tab_general:
    st.subheader("Campograma Opta 100×100 — clic para guardar coordenadas")

    # Figura (incluye puntos ya guardados)
    fig = build_pitch_figure(show_grid=grid_on, step=grid_step, clicks=st.session_state.clicks)

    # Render + captura de eventos
    st.caption("Origen (0,0) = esquina superior izquierda • X→derecha • Y→abajo")
    events = plotly_events(
        fig,
        click_event=True, select_event=False, hover_event=False,
        override_width="100%",  # ocupa ancho del contenedor
    )

    # Procesar último clic
    if events:
        px, py = float(events[-1]["x"]), float(events[-1]["y"])
        # Acotar por si el clic se va fuera del área
        px = min(100.0, max(0.0, px))
        py = min(100.0, max(0.0, py))
        if snap:
            px = snap_value(px, grid_step)
            py = snap_value(py, grid_step)

        item = {"x": round(px, 2), "y": round(py, 2), "ts": datetime.utcnow().isoformat() + "Z"}
        if accumulate:
            # Evitar duplicar el mismo evento inmediato
            if not st.session_state.clicks or st.session_state.clicks[-1] != item:
                st.session_state.clicks.append(item)
        else:
            st.session_state.clicks = [item]

    # Panel derecho con resultado
    c1, c2 = st.columns([1.2, 1])
    with c1:
        if st.session_state.clicks:
            st.success(f"Último clic → x: **{st.session_state.clicks[-1]['x']}**, y: **{st.session_state.clicks[-1]['y']}**")
            df = pd.DataFrame(st.session_state.clicks)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Descargas
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Descargar CSV", data=csv, file_name="coordenadas_opta.csv", mime="text/csv")

            js = json.dumps(st.session_state.clicks, ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button("⬇️ Descargar JSON", data=js, file_name="coordenadas_opta.json", mime="application/json")
        else:
            st.info("Hacé clic en el campo para capturar coordenadas.")

    with c2:
        st.markdown("**Opciones**")
        st.write(f"- Grilla guía: `{grid_on}` • paso: `{grid_step}`")
        st.write(f"- Snap a grilla: `{snap}`")
        st.write(f"- Acumular clics: `{accumulate}`")
        if st.session_state.clicks:
            st.caption("Tip: podés arrastrar para mover el canvas (no afecta las coords).")

# ---------------- Tabs 2 y 3 (placeholders por ahora) ---------------- #
with tab_equipos:
    st.info("Acá mostraremos métricas por equipo cuando definamos la estructura de datos.")

with tab_jugadores:
    st.info("Acá mostraremos métricas por jugador cuando definamos la estructura de datos.")