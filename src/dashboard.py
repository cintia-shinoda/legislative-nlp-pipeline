"""
dashboard.py
Dashboard para visualização de sentimento e tópicos das sessões plenárias.
Para executar: streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# ============================================================
#              CONFIGURAÇÃO DA PÁGINA
# ============================================================

# st.set_page_config() DEVE ser o primeiro comando Streamlit
# Configura título da aba, ícone e layout
st.set_page_config(
    page_title='Análise de Sentimento e Tópicos',
    layout='wide',          # Usa largura total da tela
    initial_sidebar_state='expanded'
)


# ============================================================
#                 CARREGAMENTO DE DADOS
# ============================================================

# @st.cache_data faz o Streamlit cachear o resultado desta função
# Na primeira vez, lê os arquivos. Nas próximas, usa o cache.
# Isso evita reler os Parquets toda vez que o usuário interage com o dashboard.
@st.cache_data
def load_data():
    """Carrega todos os dados processados."""
    
    data_dir = Path('data/output')
    
    # Dados principais: todos os segmentos com sentimento e tópicos
    df = pd.read_parquet(data_dir / 'all_sessions_with_topics.parquet')
    
    # Info dos tópicos (nome, contagem)
    topic_info = pd.read_parquet(data_dir / 'topic_info.parquet')
    
    # Cruzamento tópico × sentimento
    cross = pd.read_parquet(data_dir / 'topic_sentiment_cross.parquet')
    
    # Extrair data da sessão a partir do nome
    # 'Sessao_Plenaria_22_10_2025' → '2025-10-22'
    def extract_date(name):
        parts = name.split('_')
        # parts = ['Sessao', 'Plenaria', '22', '10', '2025']
        if len(parts) >= 5:
            day, month, year = parts[2], parts[3], parts[4]
            return f'{year}-{month}-{day}'
        return name
    
    df['data_sessao'] = df['sessao'].apply(extract_date)
    df['data_sessao'] = pd.to_datetime(df['data_sessao'])
    
    return df, topic_info, cross


# Carregar dados
df, topic_info, cross = load_data()


# ============================================================
#                      SIDEBAR (filtros)
# ============================================================

# st.sidebar cria elementos na barra lateral
st.sidebar.title('Filtros')

# Filtro de sessão
# st.sidebar.multiselect() cria um seletor de múltiplas opções
# sorted() + unique() lista todas as sessões únicas em ordem
sessoes_disponiveis = sorted(df['sessao'].unique())

# Formatar nomes para exibição (mais legível)
sessao_labels = {s: s.replace('Sessao_Plenaria_', '').replace('_', '/') for s in sessoes_disponiveis}

sessoes_selecionadas = st.sidebar.multiselect(
    'Sessões:',
    options=sessoes_disponiveis,
    default=sessoes_disponiveis,          # Todas selecionadas por padrão
    format_func=lambda x: sessao_labels[x]  # Mostra '22/10/2025' em vez do nome completo
)

# Filtro de sentimento
sentimentos = st.sidebar.multiselect(
    'Sentimento:',
    options=['positivo', 'neutro', 'negativo'],
    default=['positivo', 'neutro', 'negativo']
)

# Filtro de tópico (excluir outliers por padrão)
incluir_outliers = st.sidebar.checkbox('Incluir outliers (sem tópico)', value=False)

# Aplicar filtros
# .isin() verifica se cada valor está na lista fornecida
df_filtered = df[
    (df['sessao'].isin(sessoes_selecionadas)) &
    (df['sentimento'].isin(sentimentos))
].copy()

if not incluir_outliers:
    df_filtered = df_filtered[df_filtered['topic_id'] != -1]

# Sempre remover tópico #0 (catch-all genérico) dos gráficos
# Ele contém 59% dos segmentos e não tem tema definido
incluir_catchall = st.sidebar.checkbox('Incluir tópico genérico (#0)', value=False)
if not incluir_catchall:
    df_filtered = df_filtered[df_filtered['topic_id'] != 0]


# ============================================================
#                       HEADER
# ============================================================

st.title('Análise de Sentimento e Tópicos')
st.markdown('**Câmara Municipal de São Paulo** — Análise de sentimento e tópicos das sessões plenárias')
st.markdown('---')


# ============================================================
#                 MÉTRICAS GERAIS (cards)
# ============================================================

# st.columns(4) cria 4 colunas lado a lado
col1, col2, col3, col4 = st.columns(4)

# st.metric() cria um card com número grande
# É o widget ideal para KPIs (Key Performance Indicators)
with col1:
    st.metric('Sessões', len(df_filtered['sessao'].unique()))

with col2:
    st.metric('Segmentos', f'{len(df_filtered):,}')

with col3:
    st.metric('Tópicos', df_filtered[df_filtered['topic_id'] != -1]['topic_id'].nunique())

with col4:
    # Confiança média do modelo de sentimento
    avg_conf = df_filtered['confianca'].mean()
    st.metric('Confiança média', f'{avg_conf:.0%}')

st.markdown('---')


# ============================================================
#        LINHA 1: Distribuição de sentimento + Timeline
# ============================================================

col_left, col_right = st.columns(2)

# --- Gráfico de pizza: distribuição geral de sentimento ---
with col_left:
    st.subheader('Distribuição de Sentimento')
    
    # value_counts() conta ocorrências de cada sentimento
    sent_dist = df_filtered['sentimento'].value_counts()
    
    # Cores consistentes para cada sentimento
    color_map = {
        'positivo': '#2ecc71',    # Verde
        'neutro': '#95a5a6',      # Cinza
        'negativo': '#e74c3c',    # Vermelho
    }
    
    fig_pie = px.pie(
        names=sent_dist.index,
        values=sent_dist.values,
        color=sent_dist.index,
        color_discrete_map=color_map,
        hole=0.4,  # 0.4 = donut chart (anel). 0 = pizza sólida
    )
    
    # update_traces() customiza a aparência
    # textinfo: o que mostrar em cada fatia
    fig_pie.update_traces(textinfo='percent+label')
    fig_pie.update_layout(showlegend=False, height=400)
    
    st.plotly_chart(fig_pie, use_container_width=True)

# --- Gráfico de barras empilhadas: sentimento por sessão ---
with col_right:
    st.subheader('Sentimento por Sessão')
    
    # pd.crosstab() cria tabela cruzada sessão × sentimento
    # normalize='index' faz cada sessão somar 100%
    timeline = pd.crosstab(
        df_filtered['data_sessao'].dt.strftime('%d/%m'),
        df_filtered['sentimento'],
        normalize='index'
    ).round(3)
    
    # Reordenar colunas
    cols_order = [c for c in ['positivo', 'neutro', 'negativo'] if c in timeline.columns]
    timeline = timeline[cols_order]
    
    fig_timeline = px.bar(
        timeline,
        barmode='stack',          # Barras empilhadas (somam 100%)
        color_discrete_map=color_map,
        labels={'value': 'Proporção', 'index': 'Sessão'},
    )
    
    fig_timeline.update_layout(
        height=400,
        yaxis_tickformat='.0%',   # Formatar eixo Y como percentual
        legend_title_text='Sentimento',
        xaxis_title='Data da sessão',
        yaxis_title='Proporção',
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)


st.markdown('---')


# ============================================================
# LINHA 2: Heatmap Tópico × Sentimento (o CRUZAMENTO PRINCIPAL)
# ============================================================

st.subheader('Termômetro: Tópico × Sentimento')
st.markdown('*Cada linha é um tópico. As cores mostram a proporção de sentimento. Ordenado por % negativo (descendente).*')

# Recalcular cruzamento com dados filtrados (respeita os filtros da sidebar)
df_topics_only = df_filtered[df_filtered['topic_id'] != -1].copy()

if len(df_topics_only) > 0:
    cross_filtered = pd.crosstab(
        df_topics_only['topic_name'],
        df_topics_only['sentimento'],
        normalize='index'
    ).round(2)
    
    # Garantir que todas as colunas existam
    for col in ['positivo', 'neutro', 'negativo']:
        if col not in cross_filtered.columns:
            cross_filtered[col] = 0.0
    
    cross_filtered = cross_filtered[['positivo', 'neutro', 'negativo']]
    
    # Ordenar por % negativo (mais negativo no topo)
    cross_filtered = cross_filtered.sort_values('negativo', ascending=True)
    # Limitar heatmap aos top 25 tópicos por volume
    top_topics = df_topics_only['topic_name'].value_counts().head(25).index
    cross_filtered = cross_filtered[cross_filtered.index.isin(top_topics)]
    
    # Criar heatmap com Plotly
    fig_heat = go.Figure(data=go.Heatmap(
        z=cross_filtered.values,
        x=['Positivo', 'Neutro', 'Negativo'],
        y=cross_filtered.index,
        # Escala de cores: branco (0%) → vermelho (100%)
        colorscale='RdYlGn_r',   # Red-Yellow-Green reverso
        text=[[f'{v:.0%}' for v in row] for row in cross_filtered.values],
        texttemplate='%{text}',
        textfont={'size': 11},
        hovertemplate='Tópico: %{y}<br>Sentimento: %{x}<br>Proporção: %{text}<extra></extra>',
    ))
    
    fig_heat.update_layout(
        height=max(500, len(cross_filtered) * 35),  # Altura dinâmica baseada no nº de tópicos
        yaxis={'dtick': 1},        # Mostrar todos os labels no eixo Y
        xaxis_side='top',          # Labels do X no topo
    )
    
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.warning('Nenhum segmento com tópico encontrado para os filtros selecionados.')


st.markdown('---')


# ============================================================
#             LINHA 3: Top tópicos por volume
# ============================================================

st.subheader('Tópicos Mais Frequentes')

if len(df_topics_only) > 0:
    topic_counts = df_topics_only['topic_name'].value_counts().head(15)
    
    fig_bar = px.bar(
        x=topic_counts.values,
        y=topic_counts.index,
        orientation='h',           # Barras horizontais
        labels={'x': 'Segmentos', 'y': 'Tópico'},
        color=topic_counts.values,
        color_continuous_scale='Blues',
    )
    
    fig_bar.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},  # Maior no topo
        showlegend=False,
        coloraxis_showscale=False,
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)


st.markdown('---')

# ================================================
# LINHA 3.5: Tópicos por Sessão (heatmap temporal)
# ================================================

st.subheader('Tópicos por Sessão')
st.markdown('*Quantos segmentos de cada tópico aparecem em cada sessão.*')

if len(df_topics_only) > 0:
    temporal = pd.crosstab(
        df_filtered[df_filtered['topic_id'] > 0]['data_sessao'].dt.strftime('%d/%m'),
        df_filtered[df_filtered['topic_id'] > 0]['topic_name']
    )
    
    top_15 = temporal.sum().nlargest(15).index
    temporal = temporal[top_15]
    
    fig_temporal = go.Figure(data=go.Heatmap(
        z=temporal.values,
        x=temporal.columns,
        y=temporal.index,
        colorscale='YlOrRd',
        text=temporal.values,
        texttemplate='%{text}',
        textfont={'size': 10},
    ))
    
    fig_temporal.update_layout(
        height=350,
        xaxis_tickangle=-45,
        xaxis_side='top',
    )
    
    st.plotly_chart(fig_temporal, use_container_width=True)


st.markdown('---')

# ===================================================
# LINHA 3.7: Distribuição de confiança por sentimento
# ===================================================

st.subheader('Confiança do Modelo por Sentimento')

col_conf_left, col_conf_right = st.columns(2)

with col_conf_left:
    fig_conf = px.histogram(
        df_filtered,
        x='confianca',
        color='sentimento',
        color_discrete_map=color_map,
        nbins=20,
        barmode='overlay',
        opacity=0.7,
        labels={'confianca': 'Confiança', 'count': 'Segmentos'},
    )
    fig_conf.update_layout(height=350, xaxis_tickformat='.0%')
    st.plotly_chart(fig_conf, use_container_width=True)

with col_conf_right:
    # Confiança média por sentimento
    conf_by_sent = df_filtered.groupby('sentimento')['confianca'].agg(['mean', 'median', 'count'])
    conf_by_sent.columns = ['Média', 'Mediana', 'Segmentos']
    conf_by_sent = conf_by_sent.round(3)
    st.dataframe(conf_by_sent, use_container_width=True)
    
    # Segmentos de baixa confiança
    low_conf = df_filtered[df_filtered['confianca'] < 0.5]
    st.metric('Segmentos com confiança < 50%', f'{len(low_conf):,} ({len(low_conf)/len(df_filtered):.0%})')


st.markdown('---')

# ============================================================
#             LINHA 4: Explorador de segmentos
# ============================================================

st.subheader('Explorador de Segmentos')
st.markdown('*Navegue pelos segmentos individuais. Use os filtros da sidebar para refinar.*')

# Filtro adicional de tópico específico
if len(df_topics_only) > 0:
    topicos_disponiveis = ['Todos'] + sorted(df_filtered['topic_name'].dropna().unique().tolist())
    topico_selecionado = st.selectbox('Filtrar por tópico:', topicos_disponiveis)
    
    df_explorer = df_filtered.copy()
    if topico_selecionado != 'Todos':
        df_explorer = df_explorer[df_explorer['topic_name'] == topico_selecionado]
    
    # Selecionar e renomear colunas para exibição
    cols_display = {
        'sessao': 'Sessão',
        'start': 'Início (s)',
        'end': 'Fim (s)',
        'text': 'Texto',
        'sentimento': 'Sentimento',
        'confianca': 'Confiança',
        'topic_name': 'Tópico',
    }
    
    df_display = df_explorer[list(cols_display.keys())].rename(columns=cols_display)
    
    # Formatar sessão para exibição
    df_display['Sessão'] = df_display['Sessão'].str.replace('Sessao_Plenaria_', '').str.replace('_', '/')
    
    # st.dataframe() cria uma tabela interativa (ordenável, pesquisável)
    st.dataframe(
        df_display,
        use_container_width=True,
        height=400,
        column_config={
            'Confiança': st.column_config.ProgressColumn(
                min_value=0, max_value=1, format='%.0%%'
            ),
        }
    )
    
    st.caption(f'Mostrando {len(df_display):,} segmentos')

# ============================================================
#         LINHA 5: WordCloud do tópico selecionado
# ============================================================

st.subheader('Nuvem de Palavras')

if len(df_topics_only) > 0:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
    topico_wc = st.selectbox(
        'Tópico para wordcloud:',
        options=sorted(df_filtered['topic_name'].dropna().unique().tolist()),
        key='wc_topic'
    )
    
    # Juntar todos os textos limpos do tópico selecionado
    textos_topico = df_filtered[
        df_filtered['topic_name'] == topico_wc
    ]['text_clean'].str.cat(sep=' ')
    
    if textos_topico.strip():
        wc = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',        # Paleta de cores
            max_words=50,
            collocations=False,        # Evita repetir bigramas
        ).generate(textos_topico)
        
        fig_wc, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        
        # st.pyplot() renderiza figuras matplotlib no Streamlit
        st.pyplot(fig_wc)
    else:
        st.info('Sem texto limpo disponível para este tópico.')


st.markdown('---')

# ============================================================
#                      FOOTER
# ============================================================

st.markdown('---')
st.markdown(
    '*Análise de Sentimento e Tópicos | '
    'Dados: Câmara Municipal de São Paulo | '
    'Modelos: faster-whisper large-v3, cardiffnlp/xlm-roberta, BERTopic*'
)