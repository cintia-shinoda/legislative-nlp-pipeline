""""
test_pipeline.py
Script de teste end-to-end para verificar a integridade do pipeline de NLP aplicado às sessões plenárias da Câmara Municipal de São Paulo.

Para rodar: python src/test_pipeline.py
"""


from pathlib import Path
import pandas as pd
import sys


def check(condition, msg):
    status = 'OK' if condition else 'X'
    print(f'  {status} {msg}')
    return condition


def main():
    print('='*60)
    print('TESTE END-TO-END — Termômetro Legislativo')
    print('='*60)
    
    errors = 0
    
    # --- 1. Verificar arquivos de áudio ---
    print('1. Arquivos de áudio')
    audio_dir = Path('data/raw/audio')
    wavs = list(audio_dir.glob('*.wav')) if audio_dir.exists() else []
    if not check(len(wavs) >= 6, f'{len(wavs)} arquivos WAV encontrados (esperado: >= 6)'):
        errors += 1
    
    # --- 2. Verificar transcrições ---
    print('2. Transcrições')
    processed_dir = Path('data/processed')
    
    transcripts = [f for f in processed_dir.glob('Sessao_Plenaria_*.parquet')
                   if '_clean' not in f.name]
    if not check(len(transcripts) >= 6, f'{len(transcripts)} transcrições encontradas'):
        errors += 1
    
    total_segs = 0
    for f in transcripts:
        df = pd.read_parquet(f)
        total_segs += len(df)
    if not check(total_segs > 10000, f'{total_segs:,} segmentos transcritos (esperado: > 10.000)'):
        errors += 1
    
    # --- 3. Verificar pré-processamento ---
    print('3. Pré-processamento')
    clean_files = list(processed_dir.glob('*_clean.parquet'))
    if not check(len(clean_files) >= 6, f'{len(clean_files)} arquivos limpos encontrados'):
        errors += 1
    
    total_clean = 0
    for f in clean_files:
        df = pd.read_parquet(f)
        total_clean += len(df)
    if not check(total_clean > 8000, f'{total_clean:,} segmentos limpos (esperado: > 8.000)'):
        errors += 1
    if total_segs > 0:
        if not check(total_clean < total_segs, f'Redução: {total_segs:,} → {total_clean:,} ({total_clean/total_segs:.0%})'):
            errors += 1
    else:
        check(False, 'Não foi possível comparar (transcrições não encontradas)')
        errors += 1
    
    # --- 4. Verificar sentimento ---
    print('4. Sentimento')
    output_dir = Path('data/output')
    sent_files = list(output_dir.glob('*_sentiment.parquet'))
    if not check(len(sent_files) >= 6, f'{len(sent_files)} arquivos de sentimento encontrados'):
        errors += 1
    
    if sent_files:
        df_sent = pd.read_parquet(sent_files[0])
        required_cols = ['text', 'sentimento', 'confianca']
        has_cols = all(c in df_sent.columns for c in required_cols)
        if not check(has_cols, f'Colunas obrigatórias presentes: {required_cols}'):
            errors += 1
        
        sentiments = df_sent['sentimento'].unique()
        if not check(set(sentiments).issubset({'positivo', 'neutro', 'negativo'}),
                     f'Sentimentos válidos: {sorted(sentiments)}'):
            errors += 1
    
    # --- 5. Verificar tópicos ---
    print('5. Tópicos')
    topics_file = output_dir / 'all_sessions_with_topics.parquet'
    if not check(topics_file.exists(), 'Arquivo de tópicos existe'):
        errors += 1
    else:
        df_topics = pd.read_parquet(topics_file)
        n_topics = df_topics[df_topics['topic_id'] >= 0]['topic_id'].nunique()
        if not check(n_topics >= 20, f'{n_topics} tópicos encontrados (esperado: >= 20)'):
            errors += 1
        
        n_segs = len(df_topics)
        if not check(n_segs == total_clean, f'Consistência: {n_segs:,} segs no tópicos == {total_clean:,} no clean'):
            errors += 1
    
    topic_info = output_dir / 'topic_info.parquet'
    if not check(topic_info.exists(), 'topic_info.parquet existe'):
        errors += 1
    
    cross_file = output_dir / 'topic_sentiment_cross.parquet'
    if not check(cross_file.exists(), 'topic_sentiment_cross.parquet existe'):
        errors += 1
    
    # --- 6. Verificar catálogo ---
    print('6. Catálogo DuckDB')
    catalog = Path('data/catalogo.duckdb')
    if not check(catalog.exists(), 'catalogo.duckdb existe'):
        errors += 1
    
    # --- 7. Verificar scripts ---
    print('7. Scripts')
    scripts = [
        'src/download_audio.py',
        'src/catalog.py',
        'src/transcribe.py',
        'src/preprocess.py',
        'src/sentiment.py',
        'src/topics.py',
        'src/dashboard.py',
    ]
    for script in scripts:
        if not check(Path(script).exists(), f'{script}'):
            errors += 1
    
    # --- 8. Verificar documentação ---
    print('8. Documentação')
    docs = ['README.md', 'requirements.txt', 'docs/fontes.md']
    for doc in docs:
        if not check(Path(doc).exists(), f'{doc}'):
            errors += 1
    
    # --- RESUMO ---
    print('\n' + '='*60)
    if errors == 0:
        print('TODOS OS TESTES PASSARAM!')
    else:
        print(f'{errors} problema(s) encontrado(s)')
    print('='*60)
    
    return errors


if __name__ == '__main__':
    sys.exit(main())
