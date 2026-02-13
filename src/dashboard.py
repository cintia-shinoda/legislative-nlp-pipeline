"""
dashboard.py
Dashboard interativo para visualização dos resultados da análise de sentimento e tópicos.
Para executar: streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
#                 CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Termômetro Legislativo",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
#                     CARGA DE DADOS
# ============================================================
@st.cache_data
def load_data():
    """Carrega todos os dados processados."""
    df = pd.read_parquet('data/output/all_sessions_with_topics.parquet')
    topic_info = pd.read_parquet('data/output/topic_info.parquet')
    cross = pd.read_parquet('data/output/topic_sentiment_cross.parquet')
    return df, topic_info, cross

df, topic_info, cross = load_data()

# ============================================================
#                   SIDEBAR — FILTROS
# ============================================================
st.sidebar.title("Termômetro Legislativo")
st.sidebar.markdown("---")

# Filtro por sessão
sessoes = sorted(df['sessao'].unique())
sessoes_sel = st.sidebar.multiselect(
    "Sessões",
    options=sessoes,
    default=sessoes,
)

# Filtro por sentimento
sentimentos_sel = st.sidebar.multiselect(
    "Sentimento",
    options=['positivo', 'neutro', 'negativo'],
    default=['positivo', 'neutro', 'negativo'],
)

# Aplicar filtros
df_filtered = df[
    (df['sessao'].isin(sessoes_sel)) &
    (df['sentimento'].isin(sentimentos_sel))
]

# ============================================================
#               SEÇÃO 1 — MÉTRICAS GERAIS
# ============================================================
st.title("Análise de Sentimento e Tópicos")
st.markdown("Análise automatizada de sentimento e tópicos — Câmara Municipal de São Paulo")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sessões", len(sessoes_sel))
col2.metric("Segmentos", f"{len(df_filtered):,}")
col3.metric("Tópicos", len(topic_info) - 1)  # -1 exclui outliers
col4.metric("Confiança Média", f"{df_filtered['confianca'].mean():.0%}")

# ============================================================
#           SEÇÃO 2 — DISTRIBUIÇÃO DE SENTIMENTO
# ============================================================
st.header("Distribuição de Sentimento")

col_donut, col_bars = st.columns(2)

with col_donut:
    sent_counts = df_filtered['sentimento'].value_counts()
    colors = {'positivo': '#059669', 'neutro': '#6B7280', 'negativo': '#DC2626'}
    fig_donut = px.pie(
        values=sent_counts.values,
        names=sent_counts.index,
        hole=0.5,
        color=sent_counts.index,
        color_discrete_map=colors,
    )
    fig_donut.update_layout(height=400)
    st.plotly_chart(fig_donut, use_container_width=True)

with col_bars:
    # Sentimento por sessão (barras empilhadas)
    sent_by_session = df_filtered.groupby(['sessao', 'sentimento']).size().reset_index(name='count')
    fig_bars = px.bar(
        sent_by_session,
        x='sessao', y='count', color='sentimento',
        color_discrete_map=colors,
        barmode='stack',
    )
    fig_bars.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig_bars, use_container_width=True)

# ============================================================
#           SEÇÃO 3 — HEATMAP TÓPICO × SENTIMENTO
# ============================================================
st.header("Heatmap: Tópico × Sentimento")

# Top 25 tópicos (excluindo outliers)
df_topics = df_filtered[df_filtered['topic_id'] != -1]
top_topics = df_topics['topic_name'].value_counts().head(25).index.tolist()
df_top = df_topics[df_topics['topic_name'].isin(top_topics)]

cross_filtered = pd.crosstab(
    df_top['topic_name'],
    df_top['sentimento'],
    normalize='index',
).round(2)

cols_order = [c for c in ['positivo', 'neutro', 'negativo'] if c in cross_filtered.columns]
cross_filtered = cross_filtered[cols_order]

if 'negativo' in cross_filtered.columns:
    cross_filtered = cross_filtered.sort_values('negativo', ascending=True)

fig_heat = px.imshow(
    cross_filtered,
    color_continuous_scale='RdYlGn_r',
    aspect='auto',
    text_auto='.0%',
)
fig_heat.update_layout(height=700)
st.plotly_chart(fig_heat, use_container_width=True)

# ============================================================
#             SEÇÃO 4 — TÓPICOS MAIS FREQUENTES
# ============================================================
st.header("Tópicos Mais Frequentes")

topic_counts = df_topics['topic_name'].value_counts().head(15).sort_values()
fig_topics = px.bar(
    x=topic_counts.values,
    y=topic_counts.index,
    orientation='h',
    color=topic_counts.values,
    color_continuous_scale='Blues',
)
fig_topics.update_layout(height=500, showlegend=False, yaxis_title='', xaxis_title='Segmentos')
st.plotly_chart(fig_topics, use_container_width=True)

# ============================================================
#           SEÇÃO 5 — DISTRIBUIÇÃO DE CONFIANÇA
# ============================================================
st.header("Distribuição de Confiança do Modelo")

fig_conf = px.histogram(
    df_filtered, x='confianca', nbins=50,
    color='sentimento', color_discrete_map=colors,
    barmode='overlay', opacity=0.7,
)
fig_conf.update_layout(height=400, xaxis_title='Confiança', yaxis_title='Segmentos')
st.plotly_chart(fig_conf, use_container_width=True)

# ============================================================
#            SEÇÃO 6 — EXPLORADOR DE SEGMENTOS
# ============================================================
st.header("Explorador de Segmentos")

col_topic, col_sent = st.columns(2)
with col_topic:
    topic_filter = st.selectbox(
        "Filtrar por tópico",
        options=['Todos'] + sorted(df_topics['topic_name'].unique().tolist()),
    )
with col_sent:
    sent_filter = st.selectbox(
        "Filtrar por sentimento",
        options=['Todos', 'positivo', 'neutro', 'negativo'],
    )

df_explore = df_filtered.copy()
if topic_filter != 'Todos':
    df_explore = df_explore[df_explore['topic_name'] == topic_filter]
if sent_filter != 'Todos':
    df_explore = df_explore[df_explore['sentimento'] == sent_filter]

st.dataframe(
    df_explore[['text', 'sentimento', 'confianca', 'topic_name', 'sessao']]
    .sort_values('confianca', ascending=False)
    .head(100),
    use_container_width=True,
    height=400,
)
st.caption(f"Mostrando {min(100, len(df_explore))} de {len(df_explore):,} segmentos")

# ============================================================
#              SEÇÃO 7 — NUVEM DE PALAVRAS
# ============================================================
st.header("Nuvem de Palavras")

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    wc_topic = st.selectbox(
        "Tópico para WordCloud",
        options=['Todos (sem outliers)'] + sorted(df_topics['topic_name'].unique().tolist()),
        key='wc_topic',
    )

    if wc_topic == 'Todos (sem outliers)':
        wc_text = ' '.join(df_topics['text'].tolist())
    else:
        wc_text = ' '.join(df_topics[df_topics['topic_name'] == wc_topic]['text'].tolist())

    wc = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=100,
        colormap='viridis',
    ).generate(wc_text)

    fig_wc, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig_wc)

except ImportError:
    st.warning("Instale wordcloud: pip install wordcloud")

# ============================================================
#              SEÇÃO 8 — HEATMAP TEMPORAL
# ============================================================
st.header("Heatmap Temporal: Sessão × Tópico")

top10_topics = df_topics['topic_name'].value_counts().head(10).index.tolist()
df_temporal = df_topics[df_topics['topic_name'].isin(top10_topics)]

temporal_cross = pd.crosstab(
    df_temporal['sessao'],
    df_temporal['topic_name'],
)

fig_temporal = px.imshow(
    temporal_cross,
    color_continuous_scale='YlOrRd',
    aspect='auto',
    text_auto=True,
)
fig_temporal.update_layout(height=500, xaxis_tickangle=-45)
st.plotly_chart(fig_temporal, use_container_width=True)

# ============================================================
#              SEÇÃO 9 — RESUMO POR SESSÃO
# ============================================================
st.header("Resumo por Sessão")

summary = df_filtered.groupby('sessao').agg(
    segmentos=('text', 'count'),
    confianca_media=('confianca', 'mean'),
    pct_positivo=('sentimento', lambda x: (x == 'positivo').mean()),
    pct_neutro=('sentimento', lambda x: (x == 'neutro').mean()),
    pct_negativo=('sentimento', lambda x: (x == 'negativo').mean()),
).round(2)

summary.columns = ['Segmentos', 'Confiança Média', '% Positivo', '% Neutro', '% Negativo']

st.dataframe(
    summary.style.format({
        'Confiança Média': '{:.0%}',
        '% Positivo': '{:.0%}',
        '% Neutro': '{:.0%}',
        '% Negativo': '{:.0%}',
    }),
    use_container_width=True,
)