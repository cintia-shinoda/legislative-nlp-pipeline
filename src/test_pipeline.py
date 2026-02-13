"""
test_pipeline.py
Teste de sanidade end-to-end do pipeline
Verifica se todos os artefatos existem e têm valores consistentes.

Para executar: python src/test_pipeline.py
"""

from pathlib import Path
import pandas as pd
import duckdb
import sys


def check(condition, message):
    """Imprime OK ou FAIL para uma verificação."""
    status = '  OK' if condition else '  FAIL'
    print(f'{status} {message}')
    return condition


def main():
    errors = 0
    
    print('=' * 60)
    print('         TESTE END-TO-END')
    print('=' * 60)
    
   # --------------------------------------------------------
    # 1. Arquivos de áudio
    # --------------------------------------------------------
    print('1. Arquivos de áudio')
    
    audio_dir = Path('data/audio')
    if audio_dir.exists():
        wavs = list(audio_dir.glob('*.wav'))
        if not check(len(wavs) >= 6, f'{len(wavs)} arquivos WAV encontrados (esperado: >= 6)'):
            errors += 1
    else:
        check(True, 'Pasta data/audio não presente (WAVs removidos após transcrição — OK)')
    
    # --------------------------------------------------------
    #                    2. Transcrições
    # --------------------------------------------------------
    print('2. Transcrições')
    
    processed_dir = Path('data/processed')
    if processed_dir.exists():
        # Transcrições brutas: Sessao_Plenaria_DD_MM_YYYY.parquet (sem sufixo _clean ou _sentiment)
        transcripts = [
            f for f in processed_dir.glob('Sessao_Plenaria_*.parquet')
            if '_clean' not in f.name and '_sentiment' not in f.name
        ]
        if not check(len(transcripts) >= 6, f'{len(transcripts)} transcrições encontradas'):
            errors += 1
        
        # Contar segmentos totais
        total_segs = 0
        for t in transcripts:
            try:
                df = pd.read_parquet(t)
                total_segs += len(df)
            except Exception as e:
                check(False, f'Erro ao ler {t.name}: {e}')
                errors += 1
        
        if total_segs > 0:
            if not check(total_segs > 9000, f'{total_segs:,} segmentos transcritos (esperado: > 9.000)'):
                errors += 1
    else:
        check(False, 'Pasta data/processed não encontrada')
        errors += 1
    
    # --------------------------------------------------------
    #                 3. Pré-processamento
    # --------------------------------------------------------
    print('3. Pré-processamento')
    
    if processed_dir.exists():
        clean_files = list(processed_dir.glob('*_clean.parquet'))
        if not check(len(clean_files) >= 6, f'{len(clean_files)} arquivos limpos encontrados'):
            errors += 1
        
        total_clean = 0
        for c in clean_files:
            try:
                df = pd.read_parquet(c)
                total_clean += len(df)
            except Exception as e:
                check(False, f'Erro ao ler {c.name}: {e}')
                errors += 1
        
        if not check(total_clean > 8000, f'{total_clean:,} segmentos limpos (esperado: > 8.000)'):
            errors += 1
        
        if total_segs > 0:
            pct = total_clean / total_segs
            if not check(total_clean < total_segs, f'Redução: {total_segs:,} → {total_clean:,} ({pct:.0%})'):
                errors += 1
        else:
            check(False, 'Não foi possível comparar (transcrições não encontradas)')
            errors += 1
    
    # --------------------------------------------------------
    #                    4. Sentimento
    # --------------------------------------------------------
    print('4. Sentimento')
    
    output_dir = Path('data/output')
    if processed_dir.exists():
        sent_files = list(output_dir.glob('*_sentiment.parquet')) if output_dir.exists() else []
        # Também verificar em data/processed caso o pipeline salve lá
        sent_files += list(processed_dir.glob('*_sentiment.parquet'))
        
        if not check(len(sent_files) >= 6, f'{len(sent_files)} arquivos de sentimento encontrados'):
            errors += 1
        
        if sent_files:
            df_sent = pd.read_parquet(sent_files[0])
            required_cols = ['text', 'sentimento', 'confianca']
            has_cols = all(c in df_sent.columns for c in required_cols)
            if not check(has_cols, f'Colunas obrigatórias presentes: {required_cols}'):
                errors += 1
            
            if has_cols:
                sentimentos = sorted(df_sent['sentimento'].unique().tolist())
                valid = sentimentos == ['negativo', 'neutro', 'positivo']
                if not check(valid, f'Sentimentos válidos: {sentimentos}'):
                    errors += 1
    
    # --------------------------------------------------------
    #                       5. Tópicos
    # --------------------------------------------------------
    print('5. Tópicos')
    
    topics_path = Path('data/output/all_sessions_with_topics.parquet')
    if topics_path.exists():
        if not check(True, 'Arquivo de tópicos existe'):
            errors += 1
        
        df_topics = pd.read_parquet(topics_path)
        n_topics = df_topics['topic_id'].nunique() - (1 if -1 in df_topics['topic_id'].values else 0)
        
        if not check(n_topics >= 30, f'{n_topics} tópicos encontrados (esperado: >= 30)'):
            errors += 1
        
        # Consistência com dados limpos
        if not check(len(df_topics) == total_clean or abs(len(df_topics) - total_clean) < 200,
                      f'Consistência: {len(df_topics):,} segs no tópicos vs {total_clean:,} no clean'):
            errors += 1
        
        # Arquivos auxiliares
        ti_path = Path('data/output/topic_info.parquet')
        if not check(ti_path.exists(), 'topic_info.parquet existe'):
            errors += 1
        
        cross_path = Path('data/output/topic_sentiment_cross.parquet')
        if not check(cross_path.exists(), 'topic_sentiment_cross.parquet existe'):
            errors += 1
    else:
        check(False, 'all_sessions_with_topics.parquet não encontrado')
        errors += 1
    
    # --------------------------------------------------------
    #                  6. Catálogo DuckDB
    # --------------------------------------------------------
    print('6. Catálogo DuckDB')
    
    db_path = Path('data/catalogo.duckdb')
    if not check(db_path.exists(), 'catalogo.duckdb existe'):
        errors += 1
    
    # --------------------------------------------------------
    #                    7. Scripts
    # --------------------------------------------------------
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
    
    for s in scripts:
        if not check(Path(s).exists(), s):
            errors += 1
    
    # --------------------------------------------------------
    #                   8. Documentação
    # --------------------------------------------------------
    print('8. Documentação')
    
    docs = ['README.md', 'requirements.txt']
    for d in docs:
        if not check(Path(d).exists(), d):
            errors += 1
    
    # Verificar docs/fontes.md (opcional)
    fontes = Path('docs/fontes.md')
    if fontes.exists():
        check(True, 'docs/fontes.md')
    
    # --------------------------------------------------------
    #                    RESULTADO FINAL
    # --------------------------------------------------------
    print()
    print('=' * 60)
    if errors == 0:
        print('         TODOS OS TESTES PASSARAM! ')
    else:
        print(f' {errors} problema(s) encontrado(s) ')
    print('=' * 60)
    
    return errors


if __name__ == '__main__':
    sys.exit(main())