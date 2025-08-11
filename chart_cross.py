# charts_cross.py
# Funciones de gráficos para análisis de centros

import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px

def _build_funnel_df(df: pd.DataFrame, cross_type: str) -> pd.DataFrame:
    """
    Devuelve un DF con las etapas y conteos para el tipo de centro indicado.
    Requiere columnas: cross_tipo, outcome_type, shot_related, ultimo_event_name
    """
    req = {"cross_tipo", "outcome_type", "shot_related", "ultimo_event_name"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    mask_tipo = (df["cross_tipo"] == cross_type)
    total = len(df)
    n_tipo = int(mask_tipo.sum())
    n_success = int(((df["cross_tipo"] == cross_type) & (df["outcome_type"] == "Successful")).sum())
    n_shot = int(((df["cross_tipo"] == cross_type) &
                  (df["outcome_type"] == "Successful") &
                  (df["shot_related"] == "Shot Related")).sum())
    n_goal = int(((df["cross_tipo"] == cross_type) &
                  (df["outcome_type"] == "Successful") &
                  (df["shot_related"] == "Shot Related") &
                  (df["ultimo_event_name"] == "Goal")).sum())

    data = [
        ["Centros totales", total],
        [cross_type, n_tipo],
        ["Successful", n_success],
        ["Tiro asociado", n_shot],
        ["Goal", n_goal],
    ]
    df_funnel = pd.DataFrame(data, columns=["etapa", "conteo"])
    df_funnel["porcentaje_prev"] = (1 + df_funnel["conteo"].pct_change()) * 100
    df_funnel.loc[0, "porcentaje_prev"] = 100
    df_funnel["etiqueta"] = df_funnel.apply(lambda r: f"{r['etapa']} ({r['porcentaje_prev']:.1f}%)", axis=1)
    return df_funnel

def funnel_por_tipo(df_scope: pd.DataFrame, open_label="Abierto", closed_label="Cerrado"):
    """
    Construye un gráfico funnel con dos subplots (Abiertos / Cerrados) sobre df_scope.
    """
    df_abiertos = _build_funnel_df(df_scope, open_label)
    df_cerrados = _build_funnel_df(df_scope, closed_label)

    fig = make_subplots(rows=1, cols=2, subplot_titles=[open_label, closed_label])

    fig_abiertos = px.funnel(
        df_abiertos, x="conteo", y="etiqueta",
        text=df_abiertos["porcentaje_prev"].apply(lambda x: f"{x:.1f}%"),
    )
    for tr in fig_abiertos.data:
        fig.add_trace(tr, row=1, col=1)

    fig_cerrados = px.funnel(
        df_cerrados, x="conteo", y="etiqueta",
        text=df_cerrados["porcentaje_prev"].apply(lambda x: f"{x:.1f}%"),
    )
    for tr in fig_cerrados.data:
        fig.add_trace(tr, row=1, col=2)

    fig.update_traces(textposition="inside", textfont_size=14)
    fig.update_layout(title_text="Embudo de Centros por tipo", showlegend=False)
    return fig
