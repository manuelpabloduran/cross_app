# Version: 2.7
# + Bloque de "Conclusiones" al final
# + Botón "Imprimir / Guardar PDF" (usa diálogo del navegador)
# Resto: igual a v2.6 (filtros sidebar, selector zonas, gráficos, situaciones principales al final)

import os, base64
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import streamlit.components.v1 as components  # <-- para el botón de imprimir
import numpy as np

from chart_cross import (
    funnel_por_tipo,
    trayectorias_split_por_resultado,   # ← añade esta
    heatmap_flow_triptych,
    heatmap_count_effectiveness,
    triple_plot_by_zone,
)


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
ss.setdefault("conclusiones", "")  # <-- texto libre

# --- keys de filtros (para reset) ---
MULTI_KEYS = [
    "flt_team","flt_rival","flt_temporada","flt_competencia",
    "flt_final","flt_tipo","flt_pie"
]
BIN_KEYS = ["flt_chipped","flt_keypass"]

def reset_filtros_callback():
    for k in MULTI_KEYS:
        st.session_state[k] = []
    for k in BIN_KEYS:
        st.session_state[k] = None
        st.session_state[k + "_ui"] = "Todos"
    if "flt_xg_default" in st.session_state:
        st.session_state["flt_xg"] = st.session_state["flt_xg_default"]

# ---------- carga del CSV ----------

def _coerce_binary(series: pd.Series) -> pd.Series:
    """
    Normaliza una serie binaria a 0/1.
    - NaN/None/''/espacios -> 0
    - true/false, yes/no, y/n, si/sí -> 1/0
    - '1'/'0' y 1.0/0.0 -> 1/0
    Devuelve dtype Int64 (nullable).
    """
    s = series.copy()

    # Caso dtype booleano (incluye pandas BooleanDtype con NA)
    if pd.api.types.is_bool_dtype(s):
        # Si es 'boolean' (nullable), rellena NA con 0; si es 'bool' puro no hay NA
        try:
            s = s.astype("Int64").fillna(0)
        except Exception:
            s = s.astype("Int64")
        return s

    # 1) NaN/None -> 0
    s = s.where(~s.isna(), 0)

    # 2) Strings vacíos o espacios -> 0 (sólo aplica a object/string)
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        s = s.replace(r'^\s*$', 0, regex=True)

    # 3) Reemplazos de tokens comunes
    s = s.replace({
        True: 1, False: 0,
        "true": 1, "false": 0, "True": 1, "False": 0,
        "yes": 1, "no": 0, "Yes": 1, "No": 0,
        "y": 1, "n": 0, "Y": 1, "N": 0,
        "si": 1, "sí": 1, "Si": 1, "Sí": 1,
        "1": 1, "0": 0,
        1.0: 1, 0.0: 0,
    })

    # 4) A numérico; cualquier cosa rara -> NaN -> 0
    s = pd.to_numeric(s, errors="coerce").fillna(0).astype("Int64")
    return s


def load_cross_stats():
    candidates = ["cross_stats.csv", "data/cross_stats.csv", "dataset/cross_stats.csv"]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        st.error("No encontré 'cross_stats.csv' (probé en raíz, /data y /dataset).")
        return None
    df = pd.read_csv(path)

    for c in ["x", "y", "endX", "endY", "xg_corrected"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["Chipped", "KeyPass"]:
        if c in df.columns:
            df[c] = _coerce_binary(df[c])

    if "xg_corrected" in df.columns:
        df["xg_corrected"] = df["xg_corrected"].fillna(0)

    return df

# ---------- figura base (pitch con box-select) ----------
def build_fig(zone=None):
    fig = go.Figure()
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, 100], visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[0, 100], visible=False),
        dragmode="select",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
    )
    fig.add_layout_image(dict(
        source=image_data_url(IMG_PATH),
        xref="x", yref="y",
        x=0, y=0, sizex=100, sizey=100,
        xanchor="left", yanchor="bottom",
        sizing="stretch",
        opacity=1.0,
        layer="below"
    ))
    step = 1
    xs = [xx for xx in range(0, 101, step) for _ in range(0, 101, step)]
    ys = [yy for _ in range(0, 101, step) for yy in range(0, 101, step)]
    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode="markers",
        marker=dict(size=5, opacity=0.01),
        hoverinfo="skip", showlegend=False
    ))
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

# ---------- filtros generales (sidebar) ----------
FILTER_KEYS = {
    "team": "TeamName",
    "rival": "TeamRival",
    "temporada": "Temporada",
    "competencia": "Competencia",
    "finalizacion": "ultimo_event_name",
    "xg": "xg_corrected",
    "tipo": "cross_tipo",
    "pie": "cross_pie",
    "chipped": "Chipped",
    "keypass": "KeyPass",
}

def general_filter_panel(df: pd.DataFrame):
    selected = {}
    with st.sidebar.expander("Filtros generales", expanded=True):

        def ms(label, colname, key):
            if colname in df.columns:
                opts = sorted([x for x in df[colname].dropna().unique().tolist()])
                val = st.multiselect(label, options=opts, default=ss.get(key, []), key=key)
                selected[key] = (colname, val)
            else:
                st.caption(f"— columna no encontrada: **{colname}**")
                selected[key] = (None, [])

        def bin_sel(label, colname, key):
            if colname in df.columns:
                options = ["Todos", "Sí (1)", "No (0)"]
                idx_default = 0
                prev = ss.get(key, None)
                if prev == 1: idx_default = 1
                elif prev == 0: idx_default = 2
                choice = st.selectbox(label, options=options, index=idx_default, key=key+"_ui")
                val = None
                if choice == "Sí (1)": val = 1
                elif choice == "No (0)": val = 0
                selected[key] = (colname, val)
                ss[key] = val
            else:
                st.caption(f"— columna no encontrada: **{colname}**")
                selected[key] = (None, None)

        # Multiselects
        ms("Equipo",         FILTER_KEYS["team"],        "flt_team")
        ms("Rival",          FILTER_KEYS["rival"],       "flt_rival")
        ms("Temporada",      FILTER_KEYS["temporada"],   "flt_temporada")
        ms("Competencia",    FILTER_KEYS["competencia"], "flt_competencia")
        ms("Finalización",   FILTER_KEYS["finalizacion"],"flt_final")
        ms("Tipo de centro", FILTER_KEYS["tipo"],        "flt_tipo")
        ms("Pie del centro", FILTER_KEYS["pie"],         "flt_pie")

        # Binarios
        bin_sel("Chipped",   FILTER_KEYS["chipped"],     "flt_chipped")
        bin_sel("Keypass",   FILTER_KEYS["keypass"],     "flt_keypass")

        # Slider xG
        xg_col = FILTER_KEYS["xg"]
        if xg_col in df.columns:
            xg_series = df[xg_col].dropna()
            if len(xg_series):
                xg_min = float(xg_series.min()); xg_max = float(xg_series.max())
                ss.setdefault("flt_xg_default", (xg_min, xg_max))
                rng = st.slider("Rango xG", min_value=xg_min, max_value=xg_max,
                                value=ss.get("flt_xg", ss["flt_xg_default"]),
                                step=0.01, key="flt_xg")
                selected["flt_xg"] = (xg_col, rng)
            else:
                st.caption("— xg_corrected sin datos numéricos")
                selected["flt_xg"] = (None, None)
        else:
            st.caption("— columna no encontrada: **xg_corrected**")
            selected["flt_xg"] = (None, None)

        # Acciones
        col_a, col_b = st.columns(2)
        with col_a:
            st.button("Limpiar filtros", on_click=reset_filtros_callback)
        with col_b:
            new_val = st.toggle("Seleccionar Zona Análisis", value=ss.show_selector, key="toggle_selector")
            if new_val != ss.show_selector:
                ss.show_selector = new_val
                st.rerun()

    return selected

def apply_general_filters(df: pd.DataFrame, sel: dict):
    if df is None:
        return None
    out = df.copy()
    for k in ["flt_team","flt_rival","flt_temporada","flt_competencia","flt_final","flt_tipo","flt_pie"]:
        colname, values = sel.get(k, (None, []))
        if colname and values:
            out = out[out[colname].isin(values)]
    for k in ["flt_chipped", "flt_keypass"]:
        colname, val = sel.get(k, (None, None))
        if colname and val is not None:
            out = out[out[colname] == val]
    colname, rng = sel.get("flt_xg", (None, None))
    if colname and isinstance(rng, tuple):
        lo, hi = float(rng[0]), float(rng[1])
        out = out[out[colname].between(lo, hi)]
    return out

# ---------- filtros por zonas ----------
def apply_zone_filters(df: pd.DataFrame, zi: dict | None, zf: dict | None):
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

# Filtros en el sidebar
selections = general_filter_panel(df)

# Contenido principal
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

# Aplicar filtros
df_after_general = apply_general_filters(df, selections)
df_scope = apply_zone_filters(df_after_general, ss.zone_inicio, ss.zone_final)

st.subheader("Análisis Gráfico")

# 1) Funnel (Plotly)
try:
    fig_funnel = funnel_por_tipo(df_scope, open_label="Abierto", closed_label="Cerrado")
    st.plotly_chart(fig_funnel, use_container_width=True)
except Exception as e:
    st.warning(f"Funnel: {e}")

# 2) Trayectorias por resultado (dos paneles con alpha según Variable de interés)
st.markdown("##### Trayectorias por resultado")

# Opciones disponibles según columnas presentes
candidate_vars = ["xA", "xT", "xg_corrected"]
available_vars = [c for c in candidate_vars if c in df_scope.columns]

if len(available_vars) == 0:
    st.info("No encontré columnas para ponderar alpha (xA, xT o xg_corrected). Se usará opacidad uniforme.")
    selected_weight = None
else:
    default_var = "xA" if "xA" in available_vars else available_vars[0]
    selected_weight = st.selectbox(
        "Variable de interés",
        available_vars,
        index=available_vars.index(st.session_state.get("sel_weight_col", default_var))
            if st.session_state.get("sel_weight_col", default_var) in available_vars
            else available_vars.index(default_var),
        key="sel_weight_col",
        help="Elegí qué variable pondera la opacidad de las flechas en los centros exitosos."
    )

# Límite de flechas por panel (aplica a ambos)
max_default = min(3000, len(df_scope)) if len(df_scope) > 0 else 1000
upper_bound = max(100, min(10000, len(df_scope))) if len(df_scope) > 0 else 100
max_n = st.slider(
    "Máximo de flechas por panel",
    min_value=100, max_value=max(100, upper_bound),
    value=max_default if max_default >= 100 else 100,
    step=100,
    help="Aplica a ambos paneles: en Exitosos mantiene el Top N por la variable elegida; en No Exitosos toma los primeros N."
)

try:
    from chart_cross import trayectorias_split_por_resultado
    fig_tray2 = trayectorias_split_por_resultado(
        df_scope,
        weight_col=selected_weight if selected_weight else "xg_corrected",  # fallback
        alpha_min=0.06,
        alpha_max=0.70,
        alpha_unsuccess=0.10,
        max_arrows=max_n,
    )
    st.pyplot(fig_tray2, use_container_width=True)
except Exception as e:
    st.warning(f"Trayectorias (split): {e}")




# 3) Heatmap+Flow triptych (mplsoccer)
try:
    fig_trip = heatmap_flow_triptych(df_scope)
    st.pyplot(fig_trip, use_container_width=True)
except Exception as e:
    st.warning(f"Heatmap/Flow: {e}")

# 4) Conteo y % efectividad por zona (mplsoccer)
try:
    fig_ce = heatmap_count_effectiveness(df_scope)
    st.pyplot(fig_ce, use_container_width=True)
except Exception as e:
    st.warning(f"Conteo/Efectividad: {e}")

# 5) Triple plot usando zona_inicio como rectangle_limits
st.subheader("Análisis x zona de finalizaciones a partir de zona seleccionada (usa Zona Inicio)")
if ss.zone_inicio:
    try:
        rect = (ss.zone_inicio["x0"], ss.zone_inicio["x1"], ss.zone_inicio["y0"], ss.zone_inicio["y1"])
        fig_tz = triple_plot_by_zone(
            df=df_scope,
            zone_name="Centro desde zona seleccionada",
            rectangle_limits=rect,
            title="",
            modo="sum",
            bin_size=3
        )
        st.pyplot(fig_tz, use_container_width=True)
    except Exception as e:
        st.warning(f"Triple plot por zona: {e}")
else:
    st.info("Para el análisis por zona seleccioná primero una Zona de Inicio en el selector de zonas.")

st.divider()

# ---- Situaciones principales (AL FINAL) ----
st.subheader("Situaciones principales")
st.caption(f"Filas resultantes: {len(df_scope)} de {len(df)}")
if not df_scope.empty and "xg_corrected" in df_scope.columns:
    col_to_show = ['TeamName','TeamRival','fecha','Competencia','Temporada','minute','second',
                   'jugador', 'ultimo_event_name','cross_tipo','cross_pie','xA','xT','xg_corrected','Chipped','KeyPass']
    st.dataframe(df_scope[col_to_show].sort_values(by="xg_corrected", ascending=False), use_container_width=True, hide_index=True)
else:
    st.dataframe(df_scope, use_container_width=True, hide_index=True)

st.divider()

# ---- Conclusiones + Botón Imprimir/Guardar PDF ----
st.subheader("Conclusiones")
ss.conclusiones = st.text_area(
    "Escribí tus conclusiones del análisis",
    value=ss.conclusiones, height=180,
    help="Este texto se guarda en la sesión actual. Podés descargarlo o incluirlo al imprimir."
)


# CSS simple para mejorar impresión (oculta botones al imprimir)
components.html("""
    <style>
    @media print {
        button, [role="button"] { display: none !important; }
        header, footer { visibility: hidden !important; }
        /* intenta preservar colores */
        .stApp { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    }
    </style>
    <div style="text-align:right; margin-top:6px;">
        <button onclick="parent.window.print()" style="padding:8px 14px; font-size:14px;">
        🖨️ Imprimir / Guardar PDF
        </button>
    </div>
""", height=60)

# ---- Fin de la app ----
st.caption("Versión 2.7 - Análisis de Centros con filtros y zonas")