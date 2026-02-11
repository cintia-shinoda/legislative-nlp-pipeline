"""
topics.py
Extrai tópicos de todas as sessões usando BERTopic.
Para executar: python src/topics.py
"""

from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
import pandas as pd
import duckdb
from pathlib import Path
import time
from datetime import datetime
import json


def create_topic_model():
    """
    Cria e configura o modelo BERTopic.
    
    Usa UMAP e HDBSCAN como modelos externos para ter controle total
    dos hiperparâmetros (em vez de depender dos defaults do BERTopic).
    
    Retorna:
        BERTopic: modelo configurado (ainda não treinado)
    """
    print('Criando modelo BERTopic...')
    
    # UMAP: redução de dimensionalidade dos embeddings
    # random_state=42 garante reprodutibilidade
    umap_model = UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
    )
    
    # HDBSCAN: clusterização baseada em densidade
    # min_cluster_size=30: mínimo de segmentos para formar um tópico
    # min_samples=5: reduz exigência de densidade → menos outliers
    hdbscan_model = HDBSCAN(
        min_cluster_size=50,
        min_samples=5,
        metric='euclidean',
        prediction_data=True,
    )
    
    topic_model = BERTopic(
        embedding_model='paraphrase-multilingual-MiniLM-L12-v2',
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics='auto',
        language='portuguese',
        verbose=False,
    )
    
    print('Modelo criado!')
    return topic_model


def analyze_topics_all_sessions(topic_model, input_dir='data/output', output_dir='data/output'):
    """
    Treina o BERTopic em TODAS as sessões juntas.
    
    Por que juntas e não uma por uma?
    - Mais dados = embeddings melhores = tópicos mais coerentes
    - Permite comparar tópicos ENTRE sessões (mesmo tópico aparece em várias)
    - Se treinarmos separado, "Saúde" na sessão 1 pode ter ID diferente da sessão 2
    
    Parâmetros:
        topic_model: modelo BERTopic configurado
        input_dir (str): pasta com arquivos _sentiment.parquet
        output_dir (str): pasta para salvar resultados
    
    Retorna:
        dict: estatísticas da análise
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Encontrar todos os arquivos de sentimento
    parquets = sorted(Path(input_dir).glob('*_sentiment.parquet'))
    
    print(f'{len(parquets)} sessões encontradas')
    print()
    
    # Carregar e concatenar todas as sessões
    # Adicionamos uma coluna 'sessao' para saber de qual arquivo veio cada segmento
    dfs = []
    for pq in parquets:
        df = pd.read_parquet(pq)
        # Extrair nome da sessão do nome do arquivo
        # 'Sessao_Plenaria_11_11_2025_sentiment' → 'Sessao_Plenaria_11_11_2025'
        sessao_name = pq.stem.replace('_sentiment', '')
        df['sessao'] = sessao_name
        dfs.append(df)
    
    # pd.concat() empilha múltiplos DataFrames em um só
    # ignore_index=True reseta o índice (0, 1, 2, ... contínuo)
    df_all = pd.concat(dfs, ignore_index=True)
    
    print(f'Total: {len(df_all):,} segmentos de {len(parquets)} sessões')
    print()
    
    # Extrair textos para o BERTopic
    # Usamos texto original (não limpo) para embeddings melhores
    texts = df_all['text'].tolist()
    
    # Treinar o modelo
    print('Processando (embeddings → UMAP → HDBSCAN → c-TF-IDF)...')
    start = time.time()
    topics, probs = topic_model.fit_transform(texts)
    elapsed = time.time() - start
    
    print(f'Concluído em {elapsed:.0f}s')
    print()
    
    # Adicionar tópicos ao DataFrame
    df_all['topic_id'] = topics
    
    # Obter info dos tópicos
    topic_info = topic_model.get_topic_info()
    
    # Criar mapeamento de ID → nome legível
    # O BERTopic gera nomes como '5_paulo_são_cidade_jockey'
    # Vamos usar as palavras-chave como nome
    topic_names = {}
    for _, row in topic_info.iterrows():
        tid = row['Topic']
        if tid == -1:
            topic_names[tid] = 'Outlier (sem tópico)'
        else:
            # row['Name'] tem formato '5_palavra1_palavra2_...'
            # Pegamos só as palavras (sem o número)
            name_parts = row['Name'].split('_')[1:5]  # Primeiras 4 palavras
            topic_names[tid] = ' | '.join(name_parts)
    
    # Adicionar nome do tópico ao DataFrame
    # .map() substitui cada valor pela correspondência no dicionário
    df_all['topic_name'] = df_all['topic_id'].map(topic_names)
    
    # Salvar DataFrame completo
    output_path = f'{output_dir}/all_sessions_with_topics.parquet'
    df_all.to_parquet(output_path, index=False)
    
    # Salvar info dos tópicos separadamente (para o dashboard)
    topic_info_path = f'{output_dir}/topic_info.parquet'
    topic_info.to_parquet(topic_info_path, index=False)
    
    # Mostrar resumo
    print('Tópicos encontrados:')
    print()
    for _, row in topic_info.iterrows():
        tid = row['Topic']
        count = row['Count']
        name = topic_names.get(tid, 'desconhecido')
        if tid == -1:
            print(f'Outliers: {count} segmentos')
        else:
            print(f'  #{tid:>2d} ({count:>3d} segs) → {name}')
    
    # Cruzamento tópico × sentimento
    print()
    print('='*60)
    print('          CRUZAMENTO: Tópico × Sentimento          ')
    print('='*60)
    print()
    
    # Filtrar outliers para o cruzamento
    df_topics = df_all[df_all['topic_id'] != -1].copy()
    
    # pd.crosstab() cria uma tabela cruzada (como tabela dinâmica no Excel)
    # normalize='index' normaliza por linha (cada tópico soma 100%)
    cross = pd.crosstab(
        df_topics['topic_name'],
        df_topics['sentimento'],
        normalize='index'
    ).round(2)
    
    # Reordenar colunas
    cols = [c for c in ['positivo', 'neutro', 'negativo'] if c in cross.columns]
    cross = cross[cols]
    
    # Ordenar por % negativo (descendente) para ver tópicos mais negativos primeiro
    if 'negativo' in cross.columns:
        cross = cross.sort_values('negativo', ascending=False)
    
    print(cross.to_string())
    
    # Salvar cruzamento
    cross_path = f'{output_dir}/topic_sentiment_cross.parquet'
    cross.to_parquet(cross_path)
    
    print()
    print(f'Arquivos salvos:')
    print(f'   {output_path}')
    print(f'   {topic_info_path}')
    print(f'   {cross_path}')
    
    return {
        'total_segmentos': len(df_all),
        'total_topicos': len(topic_info) - 1,
        'outliers': int((df_all['topic_id'] == -1).sum()),
        'tempo_segundos': round(elapsed, 1),
    }


def update_catalog_status(db_path, status='concluido'):
    """
    Atualiza status de tópicos para todas as sessões no DuckDB.
    """
    con = duckdb.connect(db_path)
    con.execute(f"UPDATE sessoes SET status_topicos = '{status}'")
    con.close()


# ============================================================
#                    EXECUÇÃO PRINCIPAL
# ============================================================
if __name__ == '__main__':
    DB_PATH = 'data/catalogo.duckdb'
    
    print('='*60)
    print('          Extração de Tópicos — BERTopic          ')
    print(f'   Início: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('='*60)
    print()
    
    # Criar modelo
    topic_model = create_topic_model()
    print()
    
    # Processar todas as sessões juntas
    stats = analyze_topics_all_sessions(topic_model)
    
    # Atualizar catálogo
    update_catalog_status(DB_PATH)
    
    print()
    print('='*60)
    print('                         RESUMO          ')
    print('='*60)
    print(f'   Segmentos processados: {stats["total_segmentos"]:,}')
    print(f'   Tópicos encontrados: {stats["total_topicos"]}')
    print(f'   Outliers: {stats["outliers"]:,}')
    print(f'   Tempo: {stats["tempo_segundos"]:.0f}s')
    print(f'   Fim: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('='*60)