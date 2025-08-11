# main.py
import io, base64, json
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
from mplsoccer import Pitch
import pandas as pd

# ---- evento plotly (v2 -> fallback a v1) ----
try:
    from streamlit_plotly_events2 import plotly_events
except Exception:
    from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Selección de zona • Opta 100x100", page_icon="⚽", layout="wide")

# --------- Estado --------- #
if "zone" not in st.session_state:
    # guardamos una única zona: x0<=x1, y0<=y1 (coordenadas Opta)
    st.session_state.zone = None

# --------- Utils --------- #
def mpl_pitch_data_url():
    pitch = Pitch(pitch_type="opta", pitch_length=100, pitch_width=100, line_zorder=2)
    fig, _ = pitch.draw(figsize=(7.2, 5.0), tight_layout=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{b64}"

def build_fig(zone):
    """Crea figura Plotly 0..100 x 0..100 (Y invertida) + fondo y (opcional) rectángulo resaltado."""
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(range=[0, 100], constrain="domain", scaleanchor="y", scaleratio=1, visible=False),
        yaxis=dict(range=[100, 0], visible=False),  # Opta: origen arriba-izq
        margin=dict(l=10, r=10, t=10, b=10),
        dragmode="pan",
        editable=True,  # permite mover/redimensionar shapes
        newshape=dict(line_color="orange", fillcolor="rgba(255,165,0,0.28)")
    )
    # fondo de campo
    fig.add_layout_image(dict(
        source=mpl_pitch_data_url(), xref="x", yref="y",
        x=0, y=100, sizex=100, sizey=100, sizing="stretch", layer="below"
    ))
    # zona actual (si existe)
    if zone is not None:
        x0, x1, y0, y1 = zone["x0"], zone["x1"], zone["y0"], zone["y1"]
        fig.add_shape(
            type="rect",
            x0=x0, x1=x1, y0=y0, y1=y1,
            line=dict(color="orange", width=2),
            fillcolor="rgba(255,165,0,0.28)",
            layer="above",
            editable=True,
            name="zona"
        )
        # punto invisible para que existan eventos aunque no haya datos
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False))
    else:
        # traza vacía para capturar eventos
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False))
        # arrancamos en modo 'dibujar rectángulo'
        fig.update_layout(dragmode="drawrect")
    return fig

def parse_relayout(relayout):
    """
    Extrae x0,x1,y0,y1 desde un dict de plotly_relayout (edición o nuevo rectángulo).
    Resuelve casos donde y0>y1 o x0>x1 (por eje invertido).
    """
    if not relayout:
        return None
    # buscar claves tipo 'shapes[0].x0'...
    keys = [k for k in relayout.keys() if k.startswith("shapes[") and (k.endswith(".x0") or k.endswith(".x1") or k.endswith(".y0") or k.endswith(".y1"))]
    if not keys:
        return None
    # tomamos el índice de shape más grande (último editado/creado)
    idxs = sorted({int(k.split("[")[1].split("]")[0]) for k in keys})
    i = idxs[-1]
    def val(name, default=None):
        return relayout.get(f"shapes[{i}].{name}", default)
    x0, x1, y0, y1 = val("x0"), val("x1"), val("y0"), val("y1")
    if None in (x0, x1, y0, y1):
        return None
    # normalizar a x0<=x1, y0<=y1
    x_low, x_high = sorted([float(x0), float(x1)])
    y_low, y_high = sorted([float(y0), float(y1)])
    # recortar a 0..100
    def clamp(a): return max(0.0, min(100.0, a))
    return dict(x0=clamp(x_low), x1=clamp(x_high), y0=clamp(y_low), y1=clamp(y_high))

# --------- UI --------- #
tab_general, tab_equipos, tab_jugadores = st.tabs([
    "Análisis general de rendimiento",
    "Rendimiento Equipos",
    "Rendimiento Jugadores"
])

with tab_general:
    st.subheader("Seleccioná una **zona** por rectángulo (Opta 100×100)")
    st.caption("Tip: si no hay zona creada, el lienzo empieza en modo **dibujar rectángulo**. Luego podés **arrastrarlo** o **redimensionarlo**.")

    fig = build_fig(st.session_state.zone)

    # Capturamos eventos de edición/dibujo del rectángulo
    # Nota: en algunos entornos el nombre es 'plotly_relayout'; en otros, 'relayout'
    try:
        evts = plotly_events(
            fig,
            events=["plotly_relayout"],
            click_event=False, select_event=False, hover_event=False,
            override_width="100%", key="pitch_fig",
        )
    except TypeError:
        evts = plotly_events(
            fig,
            events=["relayout"],
            click_event=False, select_event=False, hover_event=False,
            override_width="100%", key="pitch_fig",
        )

    if evts:
        # `evts` es lista; tomamos el último relayoutData (dict)
        last = evts[-1]
        zone = parse_relayout(last)
        if zone is not None:
            st.session_state.zone = zone

    # Panel derecho con info/acciones
    c1, c2 = st.columns([1.3, 1])

    with c1:
        if st.session_state.zone:
            z = st.session_state.zone
            st.success(f"Zona seleccionada: x0={z['x0']:.2f}, x1={z['x1']:.2f} • y0={z['y0']:.2f}, y1={z['y1']:.2f}")
            # Descargas (CSV/JSON)
            df = pd.DataFrame([{
                "x0": round(z["x0"],2), "x1": round(z["x1"],2),
                "y0": round(z["y0"],2), "y1": round(z["y1"],2),
                "ts": datetime.utcnow().isoformat() + "Z"
            }])
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Descargar zona (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="zona_opta.csv",
                mime="text/csv"
            )
            st.download_button(
                "⬇️ Descargar zona (JSON)",
                data=json.dumps(df.iloc[0].to_dict(), ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="zona_opta.json",
                mime="application/json"
            )
        else:
            st.info("Dibujá una zona arrastrando el mouse para crear un rectángulo.")

    with c2:
        st.markdown("**Acciones**")
        if st.button("🧹 Limpiar zona"):
            st.session_state.zone = None
            st.rerun()
        st.caption(
            "- **Dibujar**: arrastrá para crear el rectángulo (modo inicial si no hay zona).\n"
            "- **Editar**: con el rectángulo creado, **arrastrá** para moverlo o **tirá de las esquinas** para redimensionar.\n"
            "- El área resaltada (naranja) es tu *validación visual*."
        )

with tab_equipos:
    st.info("(Placeholder) Acá irá el rendimiento por equipos más adelante.")

with tab_jugadores:
    st.info("(Placeholder) Acá irá el rendimiento por jugadores más adelante.")