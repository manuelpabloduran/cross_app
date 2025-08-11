# main.py
import json
from datetime import datetime
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

# ---------------- Config ---------------- #
st.set_page_config(page_title="Selección de coordenadas (Opta 100x100)", page_icon="⚽", layout="wide")

# ---------------- Helpers ---------------- #
def draw_pitch(width=900, height=600, rows=6, cols=5):
    """
    Devuelve imagen PIL de un campo + grilla rows x cols y el rectángulo 'inner'
    (área jugable) para mapear píxeles -> [0..100]x[0..100].
    """
    green = (22, 92, 22)
    green_dark = (16, 70, 16)
    white = (245, 245, 245)
    grid = (230, 230, 230)

    im = Image.new("RGB", (width, height), green)
    dr = ImageDraw.Draw(im)

    pad = 20  # padding exterior
    x0, y0, x1, y1 = pad, pad, width - pad, height - pad
    inner = (x0, y0, x1, y1)

    # Rayado césped
    stripe = 40
    for x in range(x0, x1, stripe):
        dr.rectangle([x, y0, x + stripe // 2, y1], fill=green_dark)

    # Líneas principales
    dr.rectangle([x0, y0, x1, y1], outline=white, width=4)
    dr.line([ (width//2, y0), (width//2, y1) ], fill=white, width=2)
    r = 60
    dr.ellipse([width//2 - r, height//2 - r, width//2 + r, height//2 + r], outline=white, width=2)

    # Áreas (simple estético)
    area_h = int((y1 - y0) * 0.32)
    dr.rectangle([x0, y0, x1, y0 + area_h], outline=white, width=3)        # Superior
    dr.rectangle([x0, y1 - area_h, x1, y1], outline=white, width=3)        # Inferior

    # Grilla (visual/guía)
    for r_i in range(rows):
        ry0 = y0 + (r_i) * (y1 - y0) / rows
        dr.line([ (x0, ry0), (x1, ry0) ], fill=grid, width=1)
    for c_i in range(cols):
        cx0 = x0 + (c_i) * (x1 - x0) / cols
        dr.line([ (cx0, y0), (cx0, y1) ], fill=grid, width=1)

    return im, inner

def pixel_to_opta(px, py, inner_box):
    """
    Mapea un punto en píxeles del canvas (px,py) al sistema Opta 0..100,0..100.
    (0,0) esquina superior-izquierda, x→derecha, y→abajo.
    """
    x0, y0, x1, y1 = inner_box
    if not (x0 <= px <= x1 and y0 <= py <= y1):
        return None, None
    rel_x = (px - x0) / (x1 - x0)  # 0..1
    rel_y = (py - y0) / (y1 - y0)  # 0..1
    opta_x = rel_x * 100.0
    opta_y = rel_y * 100.0
    return float(opta_x), float(opta_y)

# ---------------- Estado ---------------- #
if "clicks" not in st.session_state:
    st.session_state.clicks = []  # lista de dicts {"x": float, "y": float, "ts": iso}

# ---------------- Sidebar ---------------- #
st.sidebar.header("Parámetros del lienzo")
rows = st.sidebar.slider("Filas guía (visual)", 3, 12, 6)
cols = st.sidebar.slider("Columnas guía (visual)", 3, 12, 5)
canvas_w = st.sidebar.slider("Ancho canvas (px)", 700, 1200, 900, step=50)
canvas_h = int(canvas_w * 2/3)

st.sidebar.header("Acciones")
acumular = st.sidebar.toggle("Acumular clics", value=True)
if st.sidebar.button("🧹 Limpiar seleccionados"):
    st.session_state.clicks = []
    st.sidebar.success("Limpio.")

# ---------------- Tabs ---------------- #
tab_general, tab_equipos, tab_jugadores = st.tabs([
    "Análisis general de rendimiento",
    "Rendimiento Equipos",
    "Rendimiento Jugadores"
])

# ---------------- Tab 1: Selección de coordenadas ---------------- #
with tab_general:
    st.subheader("Campograma 100×100 (Opta) — seleccionar con un clic")
    pitch_img, inner_box = draw_pitch(width=canvas_w, height=canvas_h, rows=rows, cols=cols)

    canvas = st_canvas(
        background_image=pitch_img,
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode="point",          # sólo clics
        fill_color="rgba(255, 165, 0, 0.0)",
        stroke_color="#ffffff",
        stroke_width=2,
        display_toolbar=False,
        key="pitch_canvas",
    )

    clicked = False
    if canvas.json_data and "objects" in canvas.json_data and len(canvas.json_data["objects"]) > 0:
        last = canvas.json_data["objects"][-1]
        # en modo 'point' crea un 'circle': center ≈ (left+radius, top+radius)
        px = last.get("left", 0) + last.get("radius", 0)
        py = last.get("top", 0) + last.get("radius", 0)
        ox, oy = pixel_to_opta(px, py, inner_box)
        if ox is not None:
            clicked = True
            item = {"x": round(ox, 2), "y": round(oy, 2), "ts": datetime.utcnow().isoformat() + "Z"}
            if acumular:
                # Evita duplicar el mismo último clic si Streamlit re-renderiza
                if len(st.session_state.clicks) == 0 or st.session_state.clicks[-1] != item:
                    st.session_state.clicks.append(item)
            else:
                st.session_state.clicks = [item]

    # UI derecha: último clic + lista + descargas
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.caption("Coordenadas Opta (0..100, 0..100)")
        if clicked:
            st.success(f"Último clic → x: **{item['x']}**, y: **{item['y']}**")
        else:
            st.info("Hacé clic en el campo para capturar coordenadas.")

        # Tabla de seleccionados
        if len(st.session_state.clicks) > 0:
            df = pd.DataFrame(st.session_state.clicks)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Descargas
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Descargar CSV", data=csv, file_name="coordenadas_opta.csv", mime="text/csv")

            js = json.dumps(st.session_state.clicks, ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button("⬇️ Descargar JSON", data=js, file_name="coordenadas_opta.json", mime="application/json")
        else:
            st.caption("No hay puntos guardados todavía.")

    with col_right:
        st.markdown("**Notas**")
        st.write(
            "- El sistema es **Opta 100×100**: (0,0) arriba-izquierda, X→derecha, Y→abajo.\n"
            "- Las líneas de la grilla son **sólo guía visual**; no afectan el valor capturado.\n"
            "- Si desmarcás *Acumular clics*, la lista mantiene **sólo el último** punto."
        )

# ---------------- Tabs 2 y 3: placeholders por ahora ---------------- #
with tab_equipos:
    st.info("Esta pestaña la completamos más adelante. Ahora sólo usamos la selección de coordenadas.")

with tab_jugadores:
    st.info("Esta pestaña la completamos más adelante. Ahora sólo usamos la selección de coordenadas.")
