"""
sentiment.py
Classifica o sentimento de cada segmento transcrito.
Modelo: cardiffnlp/twitter-xlm-roberta-base-sentiment
Para executar: python src/sentiment.py
"""

from transformers import pipeline
import pandas as pd
import duckdb
from pathlib import Path
import time
from datetime import datetime


def load_sentiment_model():
    """
    Carrega o modelo de sentimento.
    
    Retorna:
        classifier: pipeline HuggingFace pronto para uso
    """
    print('Carregando modelo de sentimento...')
    
    # pipeline() encapsula todo o processo:
    #   1. Tokenizar texto (quebrar em tokens que o modelo entende)
    #   2. Passar pelo modelo neural (inferência)
    #   3. Converter saída numérica em label legível
    #
    # device=-1 força uso de CPU (compatível com qualquer máquina)
    # Para GPU NVIDIA seria device=0
    classifier = pipeline(
        'sentiment-analysis',
        model='cardiffnlp/twitter-xlm-roberta-base-sentiment',
        device=-1
    )
    
    print('Modelo carregado!')
    return classifier


def classify_batch(classifier, texts, batch_size=32):
    """
    Classifica uma lista de textos em batch (mais eficiente que um por um).
    
    O modelo processa 'batch_size' textos de cada vez na memória.
    Isso é muito mais rápido que chamar classifier() para cada texto
    individualmente, porque aproveita paralelismo interno do modelo.
    
    Parâmetros:
        classifier: pipeline HuggingFace
        texts (list): lista de strings para classificar
        batch_size (int): quantos textos processar por vez
    
    Retorna:
        list: lista de dicionários com 'label' e 'score'
    """
    results = []
    
    # Processar em blocos de batch_size
    # range(0, len, step) gera: 0, 32, 64, 96, ...
    for i in range(0, len(texts), batch_size):
        # Fatiar a lista: texts[0:32], texts[32:64], etc.
        batch = texts[i:i + batch_size]
        
        # truncation=True: corta textos maiores que 512 tokens
        #   (limite do modelo — textos longos demais causam erro sem isso)
        # padding=True: preenche textos curtos para igualar tamanho no batch
        #   (o modelo precisa que todos os inputs tenham o mesmo comprimento)
        batch_results = classifier(batch, truncation=True, padding=True)
        results.extend(batch_results)
    
    return results


def map_label(label):
    """
    Converte label do modelo (inglês) para português.
    
    O modelo retorna 'positive', 'negative', 'neutral'.
    Para o dashboard e relatório, usamos português.
    
    Parâmetros:
        label (str): label em inglês
    
    Retorna:
        str: label em português
    """
    mapping = {
        'positive': 'positivo',
        'negative': 'negativo',
        'neutral': 'neutro',
    }
    return mapping.get(label, label)


def analyze_session(classifier, parquet_path, output_dir='data/output'):
    """
    Analisa sentimento de todos os segmentos de uma sessão.
    
    Parâmetros:
        classifier: pipeline HuggingFace
        parquet_path (str): caminho do Parquet limpo (_clean.parquet)
        output_dir (str): pasta para salvar resultado
    
    Retorna:
        dict: estatísticas da análise
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extrair nome base: 'Sessao_Plenaria_11_11_2025_clean' → 'Sessao_Plenaria_11_11_2025'
    basename = Path(parquet_path).stem.replace('_clean', '')
    
    print(f'Analisando: {basename}')
    start = time.time()
    
    # Ler dados pré-processados
    df = pd.read_parquet(parquet_path)
    
    # Usar texto ORIGINAL (não o limpo) para análise de sentimento
    # Motivo: o modelo foi treinado em texto natural, não em lemas
    # O texto limpo (sem stopwords) pode confundir o modelo
    texts = df['text'].tolist()
    
    # Classificar em batch
    results = classify_batch(classifier, texts, batch_size=32)
    
    # Adicionar resultados ao DataFrame
    # Lista de dicionários → cada um com 'label' e 'score'
    df['sentimento'] = [map_label(r['label']) for r in results]
    df['confianca'] = [round(r['score'], 3) for r in results]
    
    # Salvar resultado
    output_path = f'{output_dir}/{basename}_sentiment.parquet'
    df.to_parquet(output_path, index=False)
    
    elapsed = time.time() - start
    
    # Calcular distribuição de sentimento
    # value_counts(normalize=True) retorna proporções (0-1) em vez de contagens
    dist = df['sentimento'].value_counts(normalize=True)
    
    # .get() retorna 0.0 se a chave não existir (ex: nenhum segmento positivo)
    pct_pos = dist.get('positivo', 0)
    pct_neg = dist.get('negativo', 0)
    pct_neu = dist.get('neutro', 0)
    
    # Confiança média por categoria
    conf_media = df.groupby('sentimento')['confianca'].mean()
    
    print(f'Concluído em {elapsed:.0f}s | '
          f'positivo {pct_pos:.0%} neutro {pct_neu:.0%} negativo {pct_neg:.0%} | '
          f'Confiança média: {df["confianca"].mean():.0%}')
    
    return {
        'arquivo': basename,
        'segmentos': len(df),
        'positivo': round(pct_pos, 3),
        'neutro': round(pct_neu, 3),
        'negativo': round(pct_neg, 3),
        'confianca_media': round(df['confianca'].mean(), 3),
        'tempo_segundos': round(elapsed, 1),
        'output_path': output_path,
    }


def update_catalog_status(db_path, basename, status='concluido'):
    """
    Atualiza o status de sentimento de uma sessão no DuckDB.
    """
    con = duckdb.connect(db_path)
    con.execute('''
        UPDATE sessoes 
        SET status_sentimento = ? 
        WHERE titulo = ?
    ''', [status, basename])
    con.close()


# ============================================================
#                   EXECUÇÃO PRINCIPAL
# ============================================================
if __name__ == '__main__':
    DB_PATH = 'data/catalogo.duckdb'
    PROCESSED_DIR = 'data/processed'
    OUTPUT_DIR = 'data/output'
    
    print('='*60)
    print('          Análise de Sentimento — cardiffnlp/xlm-roberta          ')
    print(f'   Início: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('='*60)
    print()
    
    # Carregar modelo
    classifier = load_sentiment_model()
    print()
    
    # Encontrar Parquets limpos (_clean.parquet)
    parquets = sorted(Path(PROCESSED_DIR).glob('*_clean.parquet'))
    total = len(parquets)
    
    print(f'{total} sessões para analisar')
    print()
    
    all_stats = []
    inicio_total = time.time()
    
    for i, pq in enumerate(parquets, 1):
        print(f'--- [{i}/{total}] ---')
        stats = analyze_session(classifier, str(pq), OUTPUT_DIR)
        
        if stats:
            all_stats.append(stats)
            update_catalog_status(DB_PATH, stats['arquivo'], 'concluido')
        print()
    
    tempo_total = time.time() - inicio_total
    
    # Resumo final
    print('='*60)
    print('          RESUMO DA ANÁLISE DE SENTIMENTO          ')
    print('='*60)
    
    total_segs = sum(s['segmentos'] for s in all_stats)
    avg_pos = sum(s['positivo'] for s in all_stats) / len(all_stats)
    avg_neu = sum(s['neutro'] for s in all_stats) / len(all_stats)
    avg_neg = sum(s['negativo'] for s in all_stats) / len(all_stats)
    avg_conf = sum(s['confianca_media'] for s in all_stats) / len(all_stats)
    
    print(f'   Sessões analisadas: {len(all_stats)}/{total}')
    print(f'   Total de segmentos: {total_segs:,}')
    print(f'   Distribuição média: positivo {avg_pos:.0%} neutro {avg_neu:.0%} negativo {avg_neg:.0%}')
    print(f'   Confiança média: {avg_conf:.0%}')
    print(f'   Tempo total: {tempo_total:.0f}s ({tempo_total/60:.1f} min)')
    print(f'   Fim: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('='*60)