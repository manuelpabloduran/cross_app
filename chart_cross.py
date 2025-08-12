# chart_cross.py
# Funciones de gráficos para análisis de centros
# Requiere: pandas, numpy, matplotlib, seaborn, mplsoccer, plotly

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mplsoccer import Pitch
import plotly.express as px
from plotly.subplots import make_subplots

# ---------------------------
# 1) Funnel por tipo (Plotly)
# ---------------------------
def funnel_por_tipo(df_scope: pd.DataFrame, open_label="Abierto", closed_label="Cerrado"):
    """
    Construye un gráfico funnel con dos subplots (Abiertos / Cerrados) sobre df_scope.
    Requiere columnas: cross_tipo, outcome_type, shot_related, ultimo_event_name
    """
    def _build_funnel_df(df: pd.DataFrame, cross_type: str) -> pd.DataFrame:
        req = {"cross_tipo", "outcome_type", "shot_related", "ultimo_event_name"}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")

        total = len(df)
        n_tipo = int((df["cross_tipo"] == cross_type).sum())
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
        dff = pd.DataFrame(data, columns=["etapa", "conteo"])
        dff["porcentaje_prev"] = (1 + dff["conteo"].pct_change()) * 100
        dff.loc[0, "porcentaje_prev"] = 100
        dff["etiqueta"] = dff.apply(lambda r: f"{r['etapa']} ({r['porcentaje_prev']:.1f}%)", axis=1)
        return dff

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
    fig.update_layout(
        title_text="Embudo de Centros por tipo",
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
    )
    return fig

# ---------------------------------------------
# 2) Trayectorias por resultado (mplsoccer/plt)
# ---------------------------------------------
def trayectorias_por_resultado(
    df: pd.DataFrame,
    x_col='x', y_col='y', end_x_col='endX', end_y_col='endY',
    outcome_col='outcome_type',
    figsize=(10, 7),
    title='Trayectorias de centros por resultado'
):
    pitch = Pitch(pitch_type='opta', pitch_color='white', line_color='black')
    fig, ax = pitch.draw(figsize=figsize, tight_layout=True)
    fig.patch.set_alpha(1.0)           # <- sin transparencia (fondo blanco)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for outcome, color in zip(['Successful', 'Unsuccessful'], ['green', 'red']):
        if outcome_col not in df.columns:
            continue
        df_out = df[df[outcome_col] == outcome]
        if not df_out.empty:
            pitch.arrows(
                df_out[x_col], df_out[y_col],
                df_out[end_x_col], df_out[end_y_col],
                color=color, ax=ax, label=outcome, alpha=0.12, width=1.5
            )

    ax.legend(loc='upper left')
    ax.set_title(title, fontsize=14, color="black")
    fig.tight_layout()
    return fig

# --------------------------------------------------------------
# 3) Triptych Heatmap + Flow (Todos / Successful / Unsuccessful)
# --------------------------------------------------------------
def heatmap_flow_triptych(
    df: pd.DataFrame,
    x_col='x', y_col='y', end_x_col='endX', end_y_col='endY',
    outcome_col='outcome_type',
    bins=(18, 16),
    figsize=(18, 10),
    facecolor='white'
):
    pitch = Pitch(pitch_type='opta', line_zorder=2, line_color='#222222', pitch_color='white')
    df_all = df
    df_succ = df[df[outcome_col] == 'Successful'] if outcome_col in df.columns else df.iloc[0:0]
    df_fail = df[df[outcome_col] == 'Unsuccessful'] if outcome_col in df.columns else df.iloc[0:0]

    # Estadísticos
    bs_all  = pitch.bin_statistic(df_all[x_col],  df_all[y_col],  statistic='count', bins=bins)
    bs_succ = pitch.bin_statistic(df_succ[x_col], df_succ[y_col], statistic='count', bins=bins)
    bs_fail = pitch.bin_statistic(df_fail[x_col], df_fail[y_col], statistic='count', bins=bins)
    vmax = max(
        np.nanmax(bs_all['statistic']) if bs_all['statistic'] is not None else 0,
        np.nanmax(bs_succ['statistic']) if bs_succ['statistic'] is not None else 0,
        np.nanmax(bs_fail['statistic']) if bs_fail['statistic'] is not None else 0
    )

    fig, axs = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor(facecolor)
    fig.patch.set_alpha(1.0)
    for ax in axs:
        ax.set_facecolor("white")

    def draw_panel(ax, df_src, bs_heatmap, title, cmap):
        pitch.draw(ax=ax)
        pitch.heatmap(bs_heatmap, ax=ax, cmap=cmap, vmin=0, vmax=vmax, edgecolors="white")
        if not df_src.empty:
            pitch.flow(
                df_src[x_col], df_src[y_col], df_src[end_x_col], df_src[end_y_col],
                color='black', arrow_type='scale', arrow_length=10,
                bins=bins, ax=ax
            )
        ax.set_title(title, color='black', fontsize=14)

    draw_panel(axs[0], df_all,  bs_all,  'Centros - Todos',      cmap='Blues')
    draw_panel(axs[1], df_succ, bs_succ, 'Centros - Exitosos',    cmap='Greens')
    draw_panel(axs[2], df_fail, bs_fail, 'Centros - No Exitosos', cmap='Reds')

    return fig

# -----------------------------------------------------------------
# 4) Heatmap (conteo) + Heatmap (% efectividad por zona) side-by-side
# -----------------------------------------------------------------
def heatmap_count_effectiveness(
    df: pd.DataFrame,
    x_col='x', y_col='y',
    outcome_col='outcome_type',  # si no existe, usa outcome_value==1
    outcome_value_col='outcome_value',
    bins=(19, 9),
    figsize=(14, 5),
    facecolor='white'
):
    pitch = Pitch(pitch_type='opta', line_zorder=2,
                  pitch_color='white', line_color="#222222")

    fig, axs = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor(facecolor)
    fig.patch.set_alpha(1.0)
    for ax in axs:
        ax.set_facecolor("white")

    # 1) Conteo absoluto
    pitch.draw(ax=axs[0])
    bs_count = pitch.bin_statistic(df[x_col], df[y_col], statistic='count', bins=bins)
    counts = bs_count['statistic'].astype(float)
    counts[counts == 0] = np.nan
    vmax_count = np.nanmax(counts)

    bs_count_plot = bs_count.copy()
    bs_count_plot['statistic'] = counts
    pcm1 = pitch.heatmap(bs_count_plot, ax=axs[0], cmap='Blues',
                         vmin=0, vmax=vmax_count, edgecolors='white')

    cx, cy = bs_count['cx'], bs_count['cy']
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            val = counts[i, j]
            if not np.isnan(val):
                pitch.annotate(f"{int(val)}", (cx[i, j], cy[i, j]),
                               ax=axs[0], ha='center', va='center',
                               color='black', fontsize=7)
    axs[0].set_title("Conteo de centros por zona", color='black')

    # 2) % Efectividad
    pitch.draw(ax=axs[1])

    if outcome_value_col in df.columns:
        mask_succ = (pd.to_numeric(df[outcome_value_col], errors="coerce") > 0)
    elif outcome_col in df.columns:
        mask_succ = (df[outcome_col] == 'Successful')
    else:
        mask_succ = pd.Series(False, index=df.index)

    bs_count2 = pitch.bin_statistic(df[x_col], df[y_col], statistic='count', bins=bins)
    bs_succ = pitch.bin_statistic(df.loc[mask_succ, x_col],
                                  df.loc[mask_succ, y_col],
                                  statistic='count', bins=bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        success_pct = (bs_succ['statistic'] / bs_count2['statistic']) * 100.0
        success_pct = np.where(bs_count2['statistic'] > 0, success_pct, np.nan)

    bs_pct = bs_count2.copy()
    bs_pct['statistic'] = success_pct

    pcm2 = pitch.heatmap(bs_pct, ax=axs[1], cmap='Greens',
                         vmin=0, vmax=100, edgecolors='white')

    cx, cy = bs_count2['cx'], bs_count2['cy']
    for i in range(success_pct.shape[0]):
        for j in range(success_pct.shape[1]):
            val = success_pct[i, j]
            if not np.isnan(val):
                pitch.annotate(f"{val:.0f}%", (cx[i, j], cy[i, j]),
                               ax=axs[1], ha='center', va='center',
                               color='black', fontsize=7)
    axs[1].set_title("Efectividad de centros x zona", color='black')

    # Colorbars
    cbar1 = fig.colorbar(pcm1, ax=axs[0], shrink=0.6)
    cbar1.set_label('Conteo', color="black")
    cbar2 = fig.colorbar(pcm2, ax=axs[1], shrink=0.6)
    cbar2.set_label('% efectividad', color="black")

    return fig

# ------------------------------------------------------------
# 5) Triple plot por zona (scatter / KDE / heatmap de valores)
# ------------------------------------------------------------
def triple_plot_by_zone(
    df: pd.DataFrame,
    zone_name: str,
    rectangle_limits: tuple,     # (x_min, x_max, y_min, y_max)
    x_col='x', y_col='y',
    end_x_col='endX', end_y_col='endY',
    value_col='xg_corrected',
    modo='sum',                  # 'sum', 'mean', etc.
    bin_size=3,
    title='',
    min_end_x=75
):
    x_min, x_max, y_min, y_max = rectangle_limits

    # Filtrar jugadas cuyo ORIGEN esté dentro del rectángulo
    df_zone = df[(df[x_col] >= x_min) & (df[x_col] <= x_max) &
                 (df[y_col] >= y_min) & (df[y_col] <= y_max)].copy()

    if end_x_col in df_zone.columns:
        df_zone = df_zone[df_zone[end_x_col] >= min_end_x]

    fig, axs = plt.subplots(1, 3, figsize=(21, 6))
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)

    pitch = Pitch('opta',
                  pitch_color="#e9f6e9",   # césped claro, sin transparencia
                  line_color='black',
                  stripe=True,
                  stripe_color='#cfe8c9')

    # 1) Scatter de finalizaciones (tamaño según valor)
    pitch.draw(ax=axs[0], tight_layout=False)
    axs[0].set_title('Puntos de Remate (Tamaño según valor)', color='black', fontsize=12)
    if not df_zone.empty and value_col in df_zone.columns:
        sizes = (pd.to_numeric(df_zone[value_col], errors="coerce").fillna(0.0) * 100).clip(lower=1.0)
        pitch.scatter(df_zone[end_x_col], df_zone[end_y_col], ax=axs[0],
                      s=sizes, color='black', edgecolors='white', alpha=0.4)

    # 2) KDE ponderado por value
    pitch.draw(ax=axs[1], tight_layout=False)
    axs[1].set_title('Mapa de calor (KDE) zonas de remate', color='black', fontsize=12)
    if not df_zone.empty and value_col in df_zone.columns:
        sns.kdeplot(
            x=df_zone[end_x_col], y=df_zone[end_y_col],
            weights=pd.to_numeric(df_zone[value_col], errors="coerce"),
            fill=True, cmap='RdYlGn', bw_adjust=0.7, levels=40,
            thresh=0.05, alpha=0.6, ax=axs[1]
        )

    # 3) Heatmap por bins (agregando value por 'modo')
    pitch.draw(ax=axs[2], tight_layout=False)
    axs[2].set_title('Acumulado de valor por zona', color='black', fontsize=12)

    x_bins = np.arange(70, 101, bin_size)
    y_bins = np.arange(0, 101, bin_size)

    df_hm = df_zone[(df_zone[end_x_col] >= 60) & (df_zone[end_x_col] <= 100)]
    vals = pd.to_numeric(df_hm[value_col], errors="coerce") if value_col in df_hm.columns else None
    bin_statistic = Pitch('opta').bin_statistic(
        df_hm[end_x_col] if end_x_col in df_hm.columns else [],
        df_hm[end_y_col] if end_y_col in df_hm.columns else [],
        values=vals if vals is not None else None,
        statistic=modo, bins=(x_bins, y_bins)
    )

    stat_vals = bin_statistic['statistic']
    if stat_vals is not None:
        stat_vals = np.array(stat_vals, dtype=float)
        stat_vals = np.where(stat_vals == 0, np.nan, stat_vals)
        bin_statistic['statistic'] = stat_vals
        pitch.heatmap(bin_statistic, ax=axs[2], cmap='Greens', edgecolors='white', linewidth=0.0001)

        # Etiquetas en bins no vacíos
        x_grid = bin_statistic['x_grid']
        y_grid = bin_statistic['y_grid']
        for i in range(stat_vals.shape[0]):
            for j in range(stat_vals.shape[1]):
                val = stat_vals[i, j]
                if val and not np.isnan(val):
                    axs[2].text(x_grid[i, j] + 1.5, y_grid[i, j] - 1.5, f'{val:.2f}',
                                color='black', fontsize=6, ha='center', va='center')

    # Dibujar rectángulo de origen (no transparente)
    for ax in axs:
        filled_rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                    linewidth=0, facecolor='white', alpha=0.85, zorder=3)
        ax.add_patch(filled_rect)
        border_rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                    linewidth=2, edgecolor='magenta', facecolor='none', zorder=4)
        ax.add_patch(border_rect)

    # Punto de media ponderada del ORIGEN por value
    if not df_zone.empty and value_col in df_zone.columns:
        weights = pd.to_numeric(df_zone[value_col], errors="coerce").fillna(1e-6).values
        mean_x = np.average(df_zone[x_col], weights=weights)
        mean_y = np.average(df_zone[y_col], weights=weights)
        for ax in axs:
            ax.scatter(mean_x, mean_y, s=100, color='white', edgecolor='black',
                       marker='o', zorder=6, label='Media ponderada')

    for ax in axs:
        ax.set_xlim(60, 100)
        ax.set_facecolor("white")

    fig.subplots_adjust(wspace=0.1, left=0.03, right=0.97, top=0.85, bottom=0.05)
    plt.suptitle(f'{zone_name} - {title}', color='black', fontsize=16)
    return fig

# -------------------------------------------------------------
# 6) Trayectorias divididas por resultado (mplsoccer)

def trayectorias_split_por_resultado(
    df: pd.DataFrame,
    x_col='x', y_col='y', end_x_col='endX', end_y_col='endY',
    outcome_col='outcome_type',
    weight_col='xA',          # columna que pondera alpha en Successful
    alpha_min=0.06,           # alpha mínimo para xA bajo
    alpha_max=0.70,           # alpha máximo para xA alto
    alpha_unsuccess=0.10,     # alpha uniforme para Unsuccessful
    figsize=(16, 6),
    max_arrows=None,          # <-- límite para ambos paneles
):
    """
    Dos paneles: Successful (izq, alpha ~ weight_col) y Unsuccessful (der, alpha fijo).
    - Successful: si max_arrows se define, toma los TOP N por weight_col (desc).
      Si weight_col no existe/no es numérico, toma los primeros N.
    - Unsuccessful: si max_arrows se define, toma los primeros N.
    Fondo blanco, textos negros.
    """
    pitch = Pitch(pitch_type='opta', pitch_color='white', line_color='black')
    fig, axs = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # fondo sólido
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    for ax in axs:
        ax.set_facecolor("white")

    # -----------------------
    # Panel Successful (izq)
    # -----------------------
    ax_succ = axs[0]
    pitch.draw(ax=ax_succ, tight_layout=False)
    ax_succ.set_title("Trayectorias - Exitosos (alpha ~ Variable de interés)", fontsize=14, color="black")

    if outcome_col in df.columns:
        succ = df[df[outcome_col] == 'Successful'].copy()
    else:
        succ = df.iloc[0:0].copy()

    # Selección top N por weight_col si procede
    use_weight = (weight_col in succ.columns)
    if use_weight:
        w = pd.to_numeric(succ[weight_col], errors='coerce')
        use_weight = w.notna().any()
    if max_arrows and max_arrows > 0 and not succ.empty:
        if use_weight:
            succ = succ.assign(_w=pd.to_numeric(succ[weight_col], errors='coerce').fillna(-np.inf))
            succ = succ.nlargest(max_arrows, columns="_w").drop(columns="_w")
        else:
            succ = succ.head(max_arrows)

    if not succ.empty:
        # calcular alphas
        if use_weight:
            w = pd.to_numeric(succ[weight_col], errors='coerce').fillna(0.0).astype(float)
            wmin, wmax = float(w.min()), float(w.max())
            rng = (wmax - wmin) if (wmax - wmin) > 1e-12 else 1.0
            alphas = alpha_min + ((w - wmin) / rng) * (alpha_max - alpha_min)
        else:
            alphas = pd.Series(alpha_max, index=succ.index)

        # para respetar alpha por flecha, dibujar en orden (menos a más opacas)
        order = np.argsort(alphas.values)
        xs = succ[x_col].values
        ys = succ[y_col].values
        xe = succ[end_x_col].values
        ye = succ[end_y_col].values
        a  = alphas.values
        for i in order:
            pitch.arrows(xs[i], ys[i], xe[i], ye[i],
                         color='green', ax=ax_succ, alpha=float(a[i]), width=1.5)

    # --------------------------
    # Panel Unsuccessful (der)
    # --------------------------
    ax_fail = axs[1]
    pitch.draw(ax=ax_fail, tight_layout=False)
    ax_fail.set_title("Trayectorias - No Exitosos", fontsize=14, color="black")

    if outcome_col in df.columns:
        fail = df[df[outcome_col] == 'Unsuccessful'].copy()
    else:
        fail = df.iloc[0:0].copy()

    if max_arrows and max_arrows > 0 and not fail.empty:
        fail = fail.head(max_arrows)

    if not fail.empty:
        pitch.arrows(
            fail[x_col], fail[y_col], fail[end_x_col], fail[end_y_col],
            color='red', ax=ax_fail, alpha=alpha_unsuccess, width=1.5
        )

    return fig