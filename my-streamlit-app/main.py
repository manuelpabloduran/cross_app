import streamlit as st
import pandas as pd

def main():
    st.title("Análisis de Centros")

    # Crear DataFrames vacíos con solo los nombres de las tablas
    st.subheader("Manual de conclusiones")
    df_conclusiones = pd.DataFrame(columns=["Columna 1", "Columna 2"])
    st.dataframe(df_conclusiones)

    st.subheader("Análisis general x zonas")
    df_zonas = pd.DataFrame(columns=["Zona", "Valor"])
    st.dataframe(df_zonas)

    st.subheader("Rendimiento Equipo")
    df_equipo = pd.DataFrame(columns=["Equipo", "Rendimiento"])
    st.dataframe(df_equipo)

    st.subheader("Rendimiento Jugador")
    df_jugador = pd.DataFrame(columns=["Jugador", "Rendimiento"])
    st.dataframe(df_jugador)

if __name__ == "__main__":
    main()