"""
TFM — Sistema integrado de análisis de sentimiento financiero y screener
inteligente para soporte a decisiones de inversión personal.

Autor: Pablo Blázquez Jiménez
"""

import json
import os
from datetime import datetime, timedelta

import feedparser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Asistente de Inversión — TFM",
    page_icon="📊",
    layout="wide",
)

ACCIONES = ["JNJ", "META", "GOOGL", "NVDA"]
ETFS = ["SPY", "MCHI", "EEM", "URTH"]
WATCHLIST = ACCIONES + ETFS

ALIAS_NOTICIAS = {
    "JNJ": ["Johnson & Johnson", "JNJ"],
    "META": ["Meta", "Facebook", "META"],
    "GOOGL": ["Google", "Alphabet", "GOOGL"],
    "NVDA": ["Nvidia", "NVDA"],
}

FUENTES_RSS = {
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "CNBC Markets": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
}

DATA_DIR = "data"


# ---------------------------------------------------------------------------
# Carga de datos históricos (cacheada, no cambia entre sesiones)
# ---------------------------------------------------------------------------

@st.cache_data
def cargar_screener_historico():
    """Carga el dataset histórico ya procesado (precios + sentimiento + score + interpretabilidad)."""
    ruta = os.path.join(DATA_DIR, "screener_completo.csv")
    df = pd.read_csv(ruta)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def cargar_umbrales_dca():
    """Carga los umbrales de percentiles por ticker usados para interpretar el score DCA."""
    ruta = os.path.join(DATA_DIR, "umbrales_dca.json")
    with open(ruta, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Datos en vivo: precios recientes
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)  # se refresca cada hora
def descargar_precios_recientes(ticker, dias=180):
    """Descarga precios recientes de un activo y calcula indicadores técnicos."""
    fin = datetime.today()
    inicio = fin - timedelta(days=dias)
    data = yf.download(ticker, start=inicio, end=fin, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    import ta
    data["RSI"] = ta.momentum.RSIIndicator(close=data["Close"], window=14).rsi()
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["Momentum_10d"] = data["Close"].pct_change(periods=10) * 100

    return data


# ---------------------------------------------------------------------------
# Datos en vivo: noticias + sentimiento (FinBERT bajo demanda)
# ---------------------------------------------------------------------------

@st.cache_resource
def cargar_finbert():
    """Carga el modelo FinBERT una sola vez por sesión del servidor."""
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")


@st.cache_data(ttl=1800)  # se refresca cada 30 min
def obtener_noticias_recientes(ticker):
    """Descarga noticias RSS recientes relevantes para un ticker."""
    if ticker not in ALIAS_NOTICIAS:
        return pd.DataFrame()

    nombres = ALIAS_NOTICIAS[ticker]
    noticias = []
    for nombre_fuente, url in FUENTES_RSS.items():
        feed = feedparser.parse(url)
        for entrada in feed.entries:
            texto = f"{entrada.get('title', '')} {entrada.get('summary', '')}"
            if any(n.lower() in texto.lower() for n in nombres):
                noticias.append(
                    {
                        "fuente": nombre_fuente,
                        "titulo": entrada.get("title", ""),
                        "link": entrada.get("link", ""),
                    }
                )
    return pd.DataFrame(noticias)


def analizar_sentimiento_noticias(df_noticias, analizador):
    """Aplica FinBERT a un dataframe de noticias en vivo."""
    if df_noticias.empty:
        return df_noticias
    resultados = analizador(df_noticias["titulo"].tolist(), truncation=True, max_length=128)
    df_noticias = df_noticias.copy()
    df_noticias["sentimiento"] = [r["label"] for r in resultados]
    df_noticias["confianza"] = [r["score"] for r in resultados]
    return df_noticias


# ---------------------------------------------------------------------------
# Lógica del score (idéntica a la usada en el desarrollo del modelo)
# ---------------------------------------------------------------------------

def calcular_score_dca_desglosado(row, sentimiento_medio=0, num_noticias=0):
    """Replica exacta de la fórmula de la señal DCA usada en el backtest histórico."""
    componentes = {"desviacion_precio": 0, "rsi": 0, "sentimiento": 0}

    if pd.notna(row["SMA_50"]) and row["SMA_50"] > 0:
        desviacion_pct = (row["Close"] - row["SMA_50"]) / row["SMA_50"] * 100
        componentes["desviacion_precio"] = float(np.clip(-desviacion_pct * 2, -40, 40))

    if pd.notna(row["RSI"]):
        if row["RSI"] < 30:
            componentes["rsi"] = 25
        elif row["RSI"] < 45:
            componentes["rsi"] = 10
        elif row["RSI"] > 70:
            componentes["rsi"] = -25
        elif row["RSI"] > 55:
            componentes["rsi"] = -10

    if num_noticias >= 2 and sentimiento_medio < -0.5:
        componentes["sentimiento"] = -20

    componentes["total"] = round(sum(componentes.values()), 2)
    return componentes


def interpretar_score(score, umbrales_ticker):
    """Traduce un score numérico a una etiqueta relativa al histórico propio del activo."""
    if score <= umbrales_ticker["p10"]:
        return "Descuento fuerte", "El precio está entre los momentos de mayor descuento de su historial."
    elif score <= umbrales_ticker["p25"]:
        return "Descuento moderado", "El precio está por debajo de su media reciente, posible oportunidad."
    elif score < umbrales_ticker["p75"]:
        return "Neutral", "El precio está en línea con su comportamiento habitual."
    elif score < umbrales_ticker["p90"]:
        return "Sobreprecio moderado", "El precio está por encima de su media reciente."
    else:
        return "Sobreprecio fuerte", "El precio está entre los momentos de mayor sobreprecio de su historial."


def grafico_desglose(componentes, titulo):
    """Genera el gráfico de barras horizontal del desglose del score."""
    etiquetas = {
        "desviacion_precio": "Desviación de precio",
        "rsi": "RSI",
        "sentimiento": "Sentimiento noticias",
    }
    nombres = [etiquetas[k] for k in ["desviacion_precio", "rsi", "sentimiento"]]
    valores = [componentes[k] for k in ["desviacion_precio", "rsi", "sentimiento"]]
    colores = ["#2ca02c" if v >= 0 else "#d62728" for v in valores]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(nombres, valores, color=colores)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(titulo)
    ax.set_xlabel("Contribución al score")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Interfaz — cabecera
# ---------------------------------------------------------------------------

st.title("📊 Asistente de Inversión Personal")
st.caption(
    "Sistema de apoyo a decisiones de inversión basado en análisis de sentimiento "
    "(FinBERT), indicadores técnicos e interpretabilidad. El sistema informa — "
    "la decisión final es siempre del usuario."
)

tab_actual, tab_historico, tab_explica = st.tabs(
    ["📈 Estado actual", "🔬 Backtest histórico", "🧩 Explicabilidad"]
)

# ---------------------------------------------------------------------------
# Pestaña 1 — Estado actual
# ---------------------------------------------------------------------------

with tab_actual:
    ticker_actual = st.selectbox("Selecciona un activo", WATCHLIST, key="ticker_actual")

    with st.spinner(f"Descargando datos recientes de {ticker_actual}..."):
        df_reciente = descargar_precios_recientes(ticker_actual)

    col_precio, col_señal = st.columns([2, 1])

    with col_precio:
        st.subheader(f"Evolución reciente — {ticker_actual}")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(df_reciente["Date"], df_reciente["Close"], label="Precio", linewidth=1.2)
        ax.plot(df_reciente["Date"], df_reciente["SMA_20"], label="Media 20d", linewidth=0.8, linestyle="--")
        ax.plot(df_reciente["Date"], df_reciente["SMA_50"], label="Media 50d", linewidth=0.8, linestyle="--")
        ax.legend()
        ax.set_ylabel("Precio")
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

        ultimo_rsi = df_reciente["RSI"].iloc[-1]
        st.metric("RSI actual (14d)", f"{ultimo_rsi:.1f}")

    with col_señal:
        st.subheader("Señal del día")

        # Sentimiento en vivo solo disponible para acciones (no ETFs)
        sentimiento_medio_hoy, num_noticias_hoy = 0, 0
        if ticker_actual in ACCIONES:
            with st.spinner("Analizando noticias recientes..."):
                df_noticias_vivo = obtener_noticias_recientes(ticker_actual)
                if not df_noticias_vivo.empty:
                    analizador = cargar_finbert()
                    df_noticias_vivo = analizar_sentimiento_noticias(df_noticias_vivo, analizador)
                    mapa_score = {"positive": 1, "neutral": 0, "negative": -1}
                    df_noticias_vivo["score_num"] = df_noticias_vivo["sentimiento"].map(mapa_score)
                    sentimiento_medio_hoy = (
                        df_noticias_vivo["score_num"] * df_noticias_vivo["confianza"]
                    ).mean()
                    num_noticias_hoy = len(df_noticias_vivo)

        fila_hoy = df_reciente.iloc[-1]
        componentes_hoy = calcular_score_dca_desglosado(
            fila_hoy, sentimiento_medio=sentimiento_medio_hoy, num_noticias=num_noticias_hoy
        )
        score_hoy = componentes_hoy["total"]

        umbrales = cargar_umbrales_dca()
        if ticker_actual in umbrales:
            etiqueta, mensaje = interpretar_score(score_hoy, umbrales[ticker_actual])
        else:
            # ETFs sin histórico de sentimiento propio: interpretación solo técnica
            etiqueta = "Sobreprecio" if score_hoy < -15 else "Descuento" if score_hoy > 15 else "Neutral"
            mensaje = "Interpretación basada solo en componentes técnicos (sin histórico de sentimiento para ETFs)."

        color_etiqueta = "🟢" if score_hoy > 10 else "🔴" if score_hoy < -10 else "🟡"
        st.markdown(f"### {color_etiqueta} {etiqueta}")
        st.write(mensaje)
        st.metric("Score", f"{score_hoy:.1f}")

        if num_noticias_hoy > 0:
            st.caption(f"Basado en {num_noticias_hoy} noticia(s) reciente(s) + indicadores técnicos.")
            with st.expander("Ver noticias analizadas"):
                for _, n in df_noticias_vivo.iterrows():
                    st.write(f"**[{n['sentimiento']}]** {n['titulo']} — *{n['fuente']}*")
        else:
            st.caption("Sin noticias recientes relevantes encontradas — señal basada solo en técnico.")

# ---------------------------------------------------------------------------
# Pestaña 2 — Backtest histórico
# ---------------------------------------------------------------------------

with tab_historico:
    df_hist = cargar_screener_historico()

    st.subheader("Evolución del score a lo largo del periodo analizado (2011–2020)")
    ticker_hist = st.selectbox("Activo", df_hist["stock"].unique(), key="ticker_hist")

    datos_ticker = df_hist[df_hist["stock"] == ticker_hist].sort_values("date")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(datos_ticker["date"], datos_ticker["score_dca"], linewidth=0.6)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Score DCA")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("¿Consigue la señal mejorar al DCA clásico?")
    st.markdown(
        """
        Se comparó una estrategia de aportación periódica fija (DCA clásico) frente a una
        variante que utiliza la señal del sistema para elegir el momento de aportación
        dentro de una ventana de días, sin usar información futura (backtest secuencial).
        """
    )

    # Tabla de resultados ya conocida del desarrollo (documentada como resultado, no recalculada en vivo)
    resumen = pd.DataFrame(
        {
            "Ticker": ["JNJ", "META", "GOOGL", "NVDA"],
            "Precio medio DCA clásico": [70.74, 70.80, 28.20, 0.645],
            "Precio medio DCA con señal": [70.66, 69.98, 28.22, 0.645],
            "Mejora (%)": [0.10, 1.15, -0.05, 0.02],
        }
    )
    st.dataframe(resumen, use_container_width=True)

    st.info(
        "La mejora es marginal y solo consistente en META (activo de mayor volatilidad). "
        "Este resultado es coherente con la literatura financiera: el *timing* rara vez "
        "bate de forma robusta a una estrategia de aportación periódica sin criterio. "
        "Por ello, el sistema se plantea como herramienta informativa y no como gestor "
        "autónomo de decisiones."
    )

    st.subheader("Importancia media de cada componente del score")
    importancia = df_hist[["aporte_precio", "aporte_rsi", "aporte_sentimiento"]].abs().mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(importancia.index, importancia.values, color="#1f77b4")
    ax.set_xlabel("Contribución media absoluta")
    plt.tight_layout()
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# Pestaña 3 — Explicabilidad
# ---------------------------------------------------------------------------

with tab_explica:
    st.subheader("Explicación de una señal concreta")
    st.caption(
        "Selecciona un activo y una fecha del periodo histórico analizado para ver "
        "el desglose exacto de por qué el sistema generó esa señal."
    )

    df_hist = cargar_screener_historico()

    col1, col2 = st.columns(2)
    with col1:
        ticker_exp = st.selectbox("Activo", df_hist["stock"].unique(), key="ticker_exp")
    with col2:
        fechas_disponibles = df_hist[df_hist["stock"] == ticker_exp]["date"].dt.date
        fecha_exp = st.selectbox("Fecha", sorted(fechas_disponibles, reverse=True), key="fecha_exp")

    fila = df_hist[(df_hist["stock"] == ticker_exp) & (df_hist["date"].dt.date == fecha_exp)]

    if not fila.empty:
        fila = fila.iloc[0]
        componentes = {
            "desviacion_precio": fila["aporte_precio"],
            "rsi": fila["aporte_rsi"],
            "sentimiento": fila["aporte_sentimiento"],
        }
        titulo = f"{ticker_exp} — {fecha_exp} | Score: {fila['score_dca']:.1f} ({fila['etiqueta_dca']})"
        fig = grafico_desglose(componentes, titulo)
        st.pyplot(fig)

        st.markdown(f"**Interpretación:** {fila['mensaje_dca']}")
        if pd.notna(fila.get("aviso_sentimiento")) and fila.get("aviso_sentimiento"):
            st.warning(fila["aviso_sentimiento"])

        with st.expander("Ver datos técnicos completos de ese día"):
            st.write(fila[["Close", "RSI", "SMA_20", "SMA_50", "Momentum_10d", "sentimiento_medio", "num_noticias"]])

st.divider()
st.caption(
    "TFM — Máster en Data Science, Big Data & Business Analytics (UCM). "
    "Fuentes de datos: Yahoo Finance (yfinance), noticias RSS y dataset histórico "
    "de Kaggle (licencia CC0-1.0). Modelo de sentimiento: FinBERT (HuggingFace, Apache 2.0)."
)
