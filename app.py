import streamlit as st
import pandas as pd
import datetime
import numpy as np
import re
from collections import Counter
from textblob import TextBlob
import requests
from io import BytesIO

# --- Configuración inicial ---
st.set_page_config(page_title="Dashboard Instagram", layout="wide")

def estilo_minimalista():
    st.markdown("""
        <style>
        /* --- FUENTES Y COLORES --- */
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
            color: #2E2E2E;
            background-color: #FAFAFA;
        }

        h1, h2, h3, h4 {
            color: #1A1A1A;
        }

        .stApp {
            background-color: #F8F9FA;
        }

        /* --- BOTONES --- */
        .stButton>button {
            background-color: #FFFFFF;
            color: #333333;
            border: 1px solid #DDDDDD;
            border-radius: 12px;
            padding: 8px 18px;
            font-weight: 500;
            transition: all 0.2s ease-in-out;
        }

        .stButton>button:hover {
            background-color: #EDEDED;
            color: black;
            border: 1px solid #AAAAAA;
        }

        /* --- CUADROS DE MÉTRICAS (CARDS) --- */
        .stMetric {
            background-color: white;
            border: 1px solid #E0E0E0;
            border-radius: 16px;
            padding: 15px;
            margin-bottom: 12px;
        }

        /* --- TABLA DE DATOS --- */
        .stDataFrame {
            border: none;
        }

        /* --- EXPANDERS --- */
        .streamlit-expanderHeader {
            font-weight: 600;
        }

        /* --- WIDGETS --- */
        .stMultiSelect, .stSlider, .stDateInput, .stSelectbox {
            background-color: white;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

# --- Cargar datos desde Google Drive ---
@st.cache_data(ttl=3600)  # Cache por 1 hora
def cargar_datos():
    # ID del archivo de Google Drive (extraído de la URL)
    file_id = "1nlPrjoOIHDzIb2cF5WTIvw_JmA3LrV9Q"
    
    # URL de descarga directa
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        # Descargar el archivo
        response = requests.get(download_url)
        response.raise_for_status()  # Verificar que la descarga fue exitosa
        
        # Leer el archivo Excel
        xls = pd.ExcelFile(BytesIO(response.content), engine="openpyxl")
        feed = pd.read_excel(xls, sheet_name="Metricas ig Feed")
        historias = pd.read_excel(xls, sheet_name="Metrica Ig Historias")

        # Procesar columnas de fecha y hora para Feed
        if 'Hora de publicación' in feed.columns:
            feed['Hora de publicación'] = pd.to_datetime(feed['Hora de publicación'], errors='coerce')
            feed['Fecha'] = feed['Hora de publicación'].dt.date
            feed['Hora'] = feed['Hora de publicación'].dt.hour

        # Procesar columnas de fecha y hora para Historias
        if 'Hora de publicación' in historias.columns:
            historias['Hora de publicación'] = pd.to_datetime(historias['Hora de publicación'], errors='coerce')
            historias['Fecha'] = historias['Hora de publicación'].dt.date
            historias['Hora'] = historias['Hora de publicación'].dt.hour

        return feed.reset_index(drop=True), historias.reset_index(drop=True)
    
    except Exception as e:
        st.error(f"Error al cargar datos desde Google Drive: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()  # Devuelve DataFrames vacíos en caso de error

# Cargar datos
feed_df, historias_df = cargar_datos()

# --- Preparar datos ---
def preparar_datos(feed_df, historias_df):
    feed_df = feed_df.copy()
    historias_df = historias_df.copy()

    # Renombrar columnas si existen
    feed_df.rename(columns={
        'Veces que se guardó': 'Guardados',
        'Veces que se compartió': 'Compartidos'
    }, inplace=True)

    historias_df.rename(columns={
        'Veces que se compartió': 'Compartidos',
        'Visitas al perfil': 'Visitas al perfil'
    }, inplace=True)

    # Conversión numérica robusta
    columnas_numericas_feed = ['Me gusta', 'Comentarios', 'Guardados', 'Alcance', 'Visualizaciones']
    columnas_numericas_hist = ['Visualizaciones', 'Alcance', 'Respuestas']

    for col in columnas_numericas_feed:
        if col in feed_df.columns:
            feed_df[col] = pd.to_numeric(feed_df[col], errors='coerce').fillna(0)

    for col in columnas_numericas_hist:
        if col in historias_df.columns:
            historias_df[col] = pd.to_numeric(historias_df[col], errors='coerce').fillna(0)

    # Calcular Interacciones y Tasa Interacción
    if all(c in feed_df.columns for c in ['Me gusta', 'Comentarios', 'Guardados']):
        feed_df['Interacciones'] = feed_df[['Me gusta', 'Comentarios', 'Guardados']].sum(axis=1)
        feed_df['Tasa Interaccion'] = feed_df['Interacciones'] / feed_df['Alcance'].replace(0, 1)

    if all(c in historias_df.columns for c in ['Respuestas', 'Alcance']):
        historias_df['Tasa Respuesta'] = historias_df['Respuestas'] / historias_df['Alcance'].replace(0, 1)

    # Día de la semana
    feed_df['DiaSemana'] = pd.to_datetime(feed_df['Fecha'], errors='coerce').dt.day_name()
    historias_df['DiaSemana'] = pd.to_datetime(historias_df['Fecha'], errors='coerce').dt.day_name()

    # Semana
    feed_df['Semana'] = pd.to_datetime(feed_df['Fecha'], errors='coerce').dt.isocalendar().week
    historias_df['Semana'] = pd.to_datetime(historias_df['Fecha'], errors='coerce').dt.isocalendar().week

    return feed_df, historias_df

# Ejecutar preparación
feed_df, historias_df = preparar_datos(feed_df, historias_df)

# --- Filtros en la barra lateral ---
st.sidebar.header("Filtros")

fecha_min = pd.to_datetime(feed_df["Fecha"]).min().date()
fecha_max = pd.to_datetime(feed_df["Fecha"]).max().date()

try:
    fecha = st.sidebar.date_input(
        "Rango de fechas",
        value=(fecha_min, fecha_max),
        min_value=fecha_min,
        max_value=fecha_max
    )
    if isinstance(fecha, tuple) and len(fecha) == 2:
        fecha_inicio, fecha_fin = fecha
    else:
        fecha_inicio = fecha_fin = fecha
except Exception:
    st.sidebar.error("Error al seleccionar fechas, usando todo el rango disponible")
    fecha_inicio = fecha_min
    fecha_fin = fecha_max

hora_inicio, hora_fin = st.sidebar.slider("Rango horario", 0, 23, (0, 23))

cuentas = st.sidebar.multiselect(
    "Cuenta", 
    options=feed_df["Nombre de la cuenta"].dropna().unique(), 
    default=feed_df["Nombre de la cuenta"].dropna().unique()
)

tipos_feed = st.sidebar.multiselect(
    "Tipo de publicación (Feed)", 
    options=feed_df["Tipo de publicación"].dropna().unique(), 
    default=feed_df["Tipo de publicación"].dropna().unique()
)

tipos_hist = st.sidebar.multiselect(
    "Tipo de publicación (Historias)", 
    options=historias_df["Tipo de publicación"].dropna().unique(), 
    default=historias_df["Tipo de publicación"].dropna().unique()
)

# --- Aplicar filtros ---
def filtrar_datos(feed_df, historias_df, fecha_inicio, fecha_fin, hora_inicio, hora_fin, cuentas, tipos_feed, tipos_hist):
    feed_df = feed_df.copy()
    historias_df = historias_df.copy()

    feed_df['Fecha'] = pd.to_datetime(feed_df['Fecha'], errors='coerce').dt.date
    historias_df['Fecha'] = pd.to_datetime(historias_df['Fecha'], errors='coerce').dt.date

    feed_filtrado = feed_df[
        (feed_df["Nombre de la cuenta"].isin(cuentas)) &
        (feed_df["Tipo de publicación"].isin(tipos_feed)) &
        (feed_df["Fecha"] >= fecha_inicio) &
        (feed_df["Fecha"] <= fecha_fin) &
        (feed_df["Hora"] >= hora_inicio) &
        (feed_df["Hora"] <= hora_fin)
    ].copy()

    historias_filtrado = historias_df[
        (historias_df["Nombre de la cuenta"].isin(cuentas)) &
        (historias_df["Tipo de publicación"].isin(tipos_hist)) &
        (historias_df["Fecha"] >= fecha_inicio) &
        (historias_df["Fecha"] <= fecha_fin) &
        (historias_df["Hora"] >= hora_inicio) &
        (historias_df["Hora"] <= hora_fin)
    ].copy()

    return feed_filtrado, historias_filtrado

# Aplicar los filtros
feed_filtrado, historias_filtrado = filtrar_datos(
    feed_df, historias_df,
    fecha_inicio, fecha_fin,
    hora_inicio, hora_fin,
    cuentas, tipos_feed, tipos_hist
)
# --- Funciones de análisis ---
def analizar_temas(df, columna_descripcion='Descripción'):
    if columna_descripcion not in df.columns:
        return []
    
    # Unir todas las descripciones
    texto = ' '.join(df[columna_descripcion].dropna().astype(str))
    
    # Limpiar texto
    texto = re.sub(r'[^\w\s]', '', texto.lower())
    
    # Eliminar stopwords en español
    stopwords = ['de', 'en', 'y', 'la', 'el', 'los', 'las', 'un', 'una', 'con', 'para', 'por', 'que', 'se', 'su']
    palabras = [palabra for palabra in texto.split() if palabra not in stopwords and len(palabra) > 3]
    
    # Contar frecuencia
    contador = Counter(palabras)
    temas_comunes = contador.most_common(10)
    
    return temas_comunes

def analizar_tono(df, columna_descripcion='Descripción'):
    if columna_descripcion not in df.columns:
        return None
    
    # Calcular polaridad promedio
    polaridades = []
    for texto in df[columna_descripcion].dropna().astype(str):
        blob = TextBlob(texto)
        polaridades.append(blob.sentiment.polarity)
    
    if polaridades:
        polaridad_promedio = sum(polaridades) / len(polaridades)
        if polaridad_promedio > 0.1:
            return "Positivo/Emocional"
        elif polaridad_promedio < -0.1:
            return "Negativo/Preocupación"
        else:
            return "Neutral/Informativo"
    return None

def generar_recomendaciones(feed_df, historias_df):
    recomendaciones = []
    
    # 1. Tipo de publicación con mejor rendimiento
    if not feed_df.empty:
        mejor_tipo_feed = feed_df.groupby('Tipo de publicación')['Tasa Interaccion'].mean().idxmax()
        rec_tipo = f"El tipo de publicación con mayor engagement en Feed es: {mejor_tipo_feed}"
        recomendaciones.append(rec_tipo)
    
    # 2. Temas con mejor rendimiento
    temas_feed = analizar_temas(feed_filtrado[feed_filtrado['Tasa Interaccion'] > feed_filtrado['Tasa Interaccion'].median()])
    if temas_feed:
        rec_temas = f"Los temas con mejor rendimiento en Feed son: {', '.join([t[0] for t in temas_feed[:3]])}"
        recomendaciones.append(rec_temas)
    
    # 3. Mejores horarios y días
    if not feed_df.empty:
        mejor_hora = feed_df.groupby('Hora')['Tasa Interaccion'].mean().idxmax()
        mejor_dia = feed_df.groupby('DiaSemana')['Tasa Interaccion'].mean().idxmax()
        rec_horario = f"El mejor horario para publicar es alrededor de las {mejor_hora}:00 y el mejor día es {mejor_dia}"
        recomendaciones.append(rec_horario)
    
    # 4. Tono de comunicación
    tono = analizar_tono(feed_filtrado[feed_filtrado['Tasa Interaccion'] > feed_filtrado['Tasa Interaccion'].median()])
    if tono:
        rec_tono = f"El estilo de comunicación que mejor funciona es: {tono}"
        recomendaciones.append(rec_tono)
    
    return recomendaciones

def generar_ideas_contenido(feed_df, historias_df):
    ideas = []
    
    # Obtener datos para la generación
    mejor_tipo = feed_df.groupby('Tipo de publicación')['Tasa Interaccion'].mean().idxmax()
    temas = analizar_temas(feed_filtrado[feed_filtrado['Tasa Interaccion'] > feed_filtrado['Tasa Interaccion'].median()])
    mejor_hora = feed_df.groupby('Hora')['Tasa Interaccion'].mean().idxmax()
    mejor_dia = feed_df.groupby('DiaSemana')['Tasa Interaccion'].mean().idxmax()
    tono = analizar_tono(feed_filtrado[feed_filtrado['Tasa Interaccion'] > feed_filtrado['Tasa Interaccion'].median()])
    
    if temas and len(temas) >= 2:
        # Idea 1
        idea1 = f"Publica una {mejor_tipo} sobre '{temas[0][0]}' usando un tono {tono}. Recomendado para {mejor_dia} a las {mejor_hora}:00"
        ideas.append(idea1)
        
        # Idea 2
        idea2 = f"Crea contenido comparando '{temas[0][0]}' y '{temas[1][0]}' en formato {mejor_tipo}. Buen horario: {mejor_dia} tarde"
        ideas.append(idea2)
        
        # Idea 3
        idea3 = f"Desarrolla una serie de 3 publicaciones sobre '{temas[2][0]}' publicando los {mejor_dia} a las {mejor_hora}:00"
        ideas.append(idea3)
    
    return ideas

def mejores_publicaciones_feed(df, top_n=1):
    if 'Tasa Interaccion' not in df.columns:
        return pd.DataFrame()
    return df.sort_values(by='Tasa Interaccion', ascending=False).head(top_n)

def mejores_historias(df, top_n=1):
    if 'Tasa Respuesta' in df.columns:
        return df.sort_values(by='Tasa Respuesta', ascending=False).head(top_n)
    elif 'Visitas al perfil' in df.columns:
        return df.sort_values(by='Visitas al perfil', ascending=False).head(top_n)
    return pd.DataFrame()

# --- Interfaz del Dashboard ---
st.title("📊 Dashboard de Instagram Avanzado")

# Crear pestañas
tab1, tab2, tab3, tab4 = st.tabs([
    "🟦 Resumen General", 
    "🟨 Publicaciones Feed", 
    "🟧 Historias", 
    "🟩 Recomendaciones"
])

with tab1:
    st.header("📌 Resumen General")
    
    # --------------------------------------------
    # 1. Cards de KPIs (con verificación de columnas)
    # --------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Card 1: Total Visualizaciones
    total_vis = 0
    if 'Visualizaciones' in feed_filtrado.columns:
        total_vis += feed_filtrado['Visualizaciones'].sum()
    if 'Visualizaciones' in historias_filtrado.columns:
        total_vis += historias_filtrado['Visualizaciones'].sum()
    col1.metric("Total Visualizaciones", f"{total_vis:,}")

    # Card 2: Alcance Total
    total_alcance = 0
    if 'Alcance' in feed_filtrado.columns:
        total_alcance += feed_filtrado['Alcance'].sum()
    if 'Alcance' in historias_filtrado.columns:
        total_alcance += historias_filtrado['Alcance'].sum()
    col2.metric("Alcance Total", f"{total_alcance:,}")

    # Card 3: Seguimientos Ganados
    total_seguimientos = 0
    if 'Seguimientos' in feed_filtrado.columns:
        total_seguimientos += feed_filtrado['Seguimientos'].sum()
    if 'Seguimientos' in historias_filtrado.columns:
        total_seguimientos += historias_filtrado['Seguimientos'].sum()
    col3.metric("Seguimientos Ganados", f"{total_seguimientos:,}")

    # Card 4: Engagement Global
    total_interacciones = 0
    if 'Interacciones' in feed_filtrado.columns:
        total_interacciones += feed_filtrado['Interacciones'].sum()
    if 'Engagement' in historias_filtrado.columns:
        total_interacciones += historias_filtrado['Engagement'].sum()
    
    engagement_global = (total_interacciones / total_alcance * 100) if total_alcance > 0 else 0
    col4.metric("Engagement Global", f"{engagement_global:.1f}%")

    # Card 5: Publicaciones Analizadas
    total_pubs = len(feed_filtrado) + len(historias_filtrado)
    col5.metric("Publicaciones Analizadas", total_pubs)

    # --------------------------------------------
    # 2. Comparativo por tipo de publicación
    # --------------------------------------------
    st.subheader("Comparativo por Tipo de Publicación")
    
    # Columnas disponibles para comparar
    posibles_metricas = {
        'Visualizaciones': 'sum',
        'Alcance': 'sum', 
        'Seguimientos': 'sum',
        'Interacciones': 'sum',
        'Tasa Interaccion': 'mean'
    }
    
    # Filtrar solo las columnas que existen
    metricas_disponibles = {
        k: v for k, v in posibles_metricas.items() 
        if k in feed_filtrado.columns
    }
    
    if metricas_disponibles and not feed_filtrado.empty:
        resumen_tipo = feed_filtrado.groupby('Tipo de publicación').agg(metricas_disponibles).reset_index()
        
        # Mostrar tabla
        st.dataframe(resumen_tipo.style.format({
            'Tasa Interaccion': '{:.2%}',
            'Visualizaciones': '{:,}',
            'Alcance': '{:,}',
            'Seguimientos': '{:,}'
        }))
        
        # Mostrar gráfico de líneas
        metricas_grafico = [m for m in metricas_disponibles if m != 'Tasa Interaccion']
        if metricas_grafico:
            st.line_chart(
                resumen_tipo.set_index('Tipo de publicación')[metricas_grafico].sort_index(),
                height=400
            )

    else:
        st.warning("No hay datos suficientes para mostrar el comparativo")

    # --------------------------------------------
    # 3. Tendencias de Engagement
    # --------------------------------------------
    st.subheader("Tendencias de Engagement")
    
    if not feed_filtrado.empty and 'Tasa Interaccion' in feed_filtrado.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Por Semana**")
            engagement_semana = feed_filtrado.groupby('Semana')['Tasa Interaccion'].mean().reset_index()
            st.line_chart(
                engagement_semana.set_index('Semana'),
                y='Tasa Interaccion',
                height=300
            )
        
        with col2:
            st.write("**Por Día de la Semana**")
            orden_dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            engagement_dia = feed_filtrado.groupby('DiaSemana')['Tasa Interaccion'].mean().reset_index()
            engagement_dia['DiaSemana'] = pd.Categorical(
                engagement_dia['DiaSemana'], 
                categories=orden_dias, 
                ordered=True
            )
            engagement_dia = engagement_dia.sort_values('DiaSemana')
            st.bar_chart(
                engagement_dia.set_index('DiaSemana'),
                y='Tasa Interaccion',
                height=300
            )
    else:
        st.warning("No hay datos de engagement disponibles")

with tab2:
    # --- Publicaciones Feed ---
    st.header("🟨 Análisis de Publicaciones Feed")
    
    if not feed_filtrado.empty:
        # Mostrar tabla con publicaciones
        st.dataframe(feed_filtrado[['Fecha', 'Hora', 'Tipo de publicación', 'Visualizaciones', 
                                  'Alcance', 'Interacciones', 'Tasa Interaccion']])
        
        # Gráfico de evolución
        st.subheader("Evolución de Métricas")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Alcance por Fecha**")
            alcance_fecha = feed_filtrado.groupby('Fecha')['Alcance'].sum().reset_index()
            st.line_chart(alcance_fecha.set_index('Fecha'))
        
        with col2:
            st.write("**Interacciones por Fecha**")
            interacciones_fecha = feed_filtrado.groupby('Fecha')['Interacciones'].sum().reset_index()
            st.line_chart(interacciones_fecha.set_index('Fecha'))
        
        # Análisis por hora
        st.subheader("Desempeño por Hora del Día")
        hora_metricas = feed_filtrado.groupby('Hora').agg({
            'Alcance': 'mean',
            'Interacciones': 'mean',
            'Tasa Interaccion': 'mean'
        }).reset_index()
        
        st.bar_chart(hora_metricas.set_index('Hora'))
    else:
        st.warning("No hay datos de feed con los filtros seleccionados")

with tab3:
    # --- Historias ---
    st.header("🟧 Análisis de Historias")
    
    if not historias_filtrado.empty:
        # Mostrar tabla con historias
        st.dataframe(historias_filtrado[['Fecha', 'Hora', 'Visualizaciones', 'Alcance', 
                                       'Visitas al perfil', 'Respuestas', 'Tasa Respuesta']])
        
        # Gráfico de visualizaciones por fecha
        st.subheader("Visualizaciones de Historias por Fecha")
        vis_hist = historias_filtrado.groupby('Fecha')['Visualizaciones'].sum().reset_index()
        st.line_chart(vis_hist.set_index('Fecha'))
        
        # Análisis de engagement
        st.subheader("Análisis de Engagement")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Tasa de Respuesta por Hora**")
            resp_hora = historias_filtrado.groupby('Hora')['Tasa Respuesta'].mean().reset_index()
            st.bar_chart(resp_hora.set_index('Hora'))
        
        with col2:
            st.write("**Visitas al Perfil por Día**")
            visitas_dia = historias_filtrado.groupby('DiaSemana')['Visitas al perfil'].sum().reset_index()
            st.bar_chart(visitas_dia.set_index('DiaSemana'))
    else:
        st.warning("No hay datos de historias con los filtros seleccionados")

with tab4:
    # --- Recomendaciones Automáticas ---
    st.header("🟩 Recomendaciones Estratégicas")
    
    # 1. Recomendaciones por datos
    st.subheader("📊 Recomendaciones Basadas en Datos")
    recomendaciones = generar_recomendaciones(feed_filtrado, historias_filtrado)
    
    if recomendaciones:
        for rec in recomendaciones:
            st.info(rec)
    else:
        st.warning("No hay suficientes datos para generar recomendaciones")
    
    # 2. Recomendaciones estratégicas
    st.subheader("🧠 Recomendaciones Estratégicas")
    
    temas_comunes = analizar_temas(feed_filtrado)
    tono_comunicacion = analizar_tono(feed_filtrado)
    
    if temas_comunes:
        st.write(f"**Temas más frecuentes:** {', '.join([t[0] for t in temas_comunes[:5]])}")
    
    if tono_comunicacion:
        st.write(f"**Estilo de comunicación predominante:** {tono_comunicacion}")
    
    if 'Descripción' in feed_filtrado.columns:
        st.write("**Ejemplo de recomendación estratégica:**")
        st.info("""
        Los contenidos que presentan preguntas directas y tocan temas emocionales 
        están generando más interacción. Recomendamos mantener ese estilo y combinarlo 
        con llamados a la acción suaves como 'Contáctanos para más información'.
        """)
    
    # 3. Mix generador de contenido
    st.subheader("🧪 Mix Generador de Contenido Estratégico")
    ideas = generar_ideas_contenido(feed_filtrado, historias_filtrado)
    
    if ideas:
        st.success("**Ideas de contenido para el próximo mes:**")
        for i, idea in enumerate(ideas[:5], 1):
            st.write(f"{i}. {idea}")
    else:
        st.warning("No hay suficientes datos para generar ideas de contenido")
    
    # Configuración avanzada (placeholder para futura implementación)
    with st.expander("⚙️ Configuración Avanzada"):
        st.write("**Opciones futuras:**")
        st.checkbox("Usar mis propias palabras clave", False)
        st.selectbox("Frecuencia de recomendaciones", ["Mensual", "Semanal"])
        st.slider("Número de recomendaciones", 3, 10, 5)
    # 4. Publicaciones recomendadas como ejemplo a replicar
st.subheader("📌 Publicaciones con Mejores Resultados")

# Mostrar ejemplos del Feed
mejores_feed = mejores_publicaciones_feed(feed_filtrado, top_n=1)
if not mejores_feed.empty:
    st.markdown("**🔁 Repite este tipo de publicación en FEED:**")
    for _, row in mejores_feed.iterrows():
        st.write(f"📅 {row['Fecha']} - 🕒 {row['Hora']}h - 🧩 Tipo: {row['Tipo de publicación']}")
        st.write(f"💬 Descripción: {row['Descripción'][:250]}...")
        st.write(f"📈 Tasa de interacción: {row['Tasa Interaccion']:.2%}")
        st.markdown("---")

# Mostrar ejemplos de Historias
mejores_hist = mejores_historias(historias_filtrado, top_n=1)
if not mejores_hist.empty:
    st.markdown("**🎯 Repite este estilo en HISTORIAS:**")
    for _, row in mejores_hist.iterrows():
        st.write(f"📅 {row['Fecha']} - 🕒 {row['Hora']}h - 🧩 Tipo: {row['Tipo de publicación']}")
        st.write(f"📊 Visitas al perfil: {row.get('Visitas al perfil', 0):,.0f}")
        st.write(f"📨 Respuestas: {row.get('Respuestas', 0):,.0f}")
        if 'Tasa Respuesta' in row:
            st.write(f"📈 Tasa de respuesta: {row['Tasa Respuesta']:.2%}")
        st.markdown("---")