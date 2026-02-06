"""
transcribe.py
Transcreve áudios WAV usando faster-whisper (modelo large-v3).
Salva as transcrições em Parquet + atualiza status no DuckDB.
Para rodar: python src/transcribe.py
"""

# faster_whisper: wrapper otimizado do Whisper usando CTranslate2
# Até 4x mais rápido que o Whisper original da OpenAI
from faster_whisper import WhisperModel

# organizar os segmentos transcritos em tabelas
import pandas as pd

# Atualizar o status de cada sessão após transcrição
import duckdb

# pathlib.Path: manipulação de caminhos de arquivos
from pathlib import Path

# time: para medir duração de cada transcrição
import time

# datetime: para registrar timestamps no log
from datetime import datetime

# json: para salvar metadados da transcrição
import json


def format_time(seconds):
    """
    Converte segundos em formato legível HhMMmSSs.
    
    Exemplo:
        format_time(3725) → '1h02m05s'
        format_time(45) → '0h00m45s'
    
    Parâmetros:
        seconds (float): tempo em segundos
    
    Retorna:
        str: tempo formatado
    """
    h = int(seconds // 3600)          # Horas inteiras
    m = int((seconds % 3600) // 60)   # Minutos restantes
    s = int(seconds % 60)             # Segundos restantes
    return f'{h}h{m:02d}m{s:02d}s'


def transcribe_audio(model, audio_path, output_dir='data/processed'):
    """
    Transcreve um arquivo de áudio e salva o resultado em Parquet.
    
    Parâmetros:
        model: instância do WhisperModel já carregada
        audio_path (str): caminho do arquivo WAV
        output_dir (str): pasta onde salvar a transcrição
    
    Retorna:
        dict: metadados da transcrição (duração, segmentos, arquivo de saída)
              ou None se falhar
    """
    
    # Criar pasta de saída se não existir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Nome base do arquivo (sem extensão)
    # 'data/raw/audio/Sessao_Plenaria_04_11_2025.wav' → 'Sessao_Plenaria_04_11_2025'
    basename = Path(audio_path).stem
    
    # Caminho do arquivo de saída (.parquet)
    output_path = f'{output_dir}/{basename}.parquet'
    
    print(f'Transcrevendo: {basename}')
    start = time.time()
    
    try:
        # Transcrever o áudio
        # segments: gerador que produz segmentos sob demanda
        # info: metadados da transcrição (idioma detectado, duração, etc.)
        segments, info = model.transcribe(
            audio_path,
            language='pt',       # Forçar português
            beam_size=5,         # 5 hipóteses em paralelo (padrão recomendado)
            vad_filter=True,     # Pular silêncios automaticamente
            vad_parameters=dict(
                min_silence_duration_ms=500  # Silêncio mínimo para considerar pausa (0.5s)
            )
        )
        
        # Coletar todos os segmentos em uma lista de dicionários
        # Cada segmento tem: início, fim e texto
        rows = []
        for seg in segments:
            rows.append({
                'start': round(seg.start, 2),   # Timestamp início (segundos)
                'end': round(seg.end, 2),        # Timestamp fim (segundos)
                'text': seg.text.strip(),        # Texto transcrito (sem espaços extras)
            })
        
        elapsed = time.time() - start
        
        # Criar DataFrame e salvar como Parquet
        df = pd.DataFrame(rows)
        df.to_parquet(output_path, index=False)
        
        # index=False: não salva o índice numérico do Pandas no arquivo
        # (economiza espaço e evita coluna inútil ao ler depois)
        
        # Calcular estatísticas
        audio_duration = rows[-1]['end'] if rows else 0  # Duração aproximada do áudio
        speed = audio_duration / elapsed if elapsed > 0 else 0
        
        print(f'Concluído em {format_time(elapsed)} | '
              f'{len(rows)} segmentos | '
              f'Velocidade: {speed:.1f}x | '
              f'Salvo: {output_path}')
        
        return {
            'arquivo': basename,
            'segmentos': len(rows),
            'duracao_transcricao': round(elapsed, 1),
            'output_path': output_path,
        }
        
    except Exception as e:
        elapsed = time.time() - start
        print(f'Erro após {format_time(elapsed)}: {e}')
        return None


def update_catalog_status(db_path, basename, status='concluido'):
    """
    Atualiza o status de transcrição de uma sessão no DuckDB.
    
    Parâmetros:
        db_path (str): caminho do banco DuckDB
        basename (str): nome do arquivo (sem extensão) para identificar a sessão
        status (str): novo status ('concluido', 'erro', 'pendente')
    """
    con = duckdb.connect(db_path)
    
    # UPDATE modifica linhas existentes na tabela
    # SET define o novo valor da coluna
    # WHERE filtra qual linha modificar (pelo título)
    con.execute('''
        UPDATE sessoes 
        SET status_transcricao = ? 
        WHERE titulo = ?
    ''', [status, basename])
    
    con.close()


# ============================================================
#                  EXECUÇÃO PRINCIPAL
# ============================================================
if __name__ == '__main__':
    DB_PATH = 'data/catalogo.duckdb'
    AUDIO_DIR = 'data/raw/audio'
    OUTPUT_DIR = 'data/processed'
    MODEL_SIZE = 'large-v3'
    
    print('='*60)
    print('Transcrição em Batch — faster-whisper')
    print(f'   Modelo: {MODEL_SIZE}')
    print(f'   Device: cpu | Compute: int8')
    print(f'   Início: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('='*60)
    

    print()
    print('Carregando modelo...')
    model = WhisperModel(MODEL_SIZE, device='cpu', compute_type='int8')
    print('Modelo carregado!')
    print()
    
    # Listar todos os áudios WAV
    audio_files = sorted(Path(AUDIO_DIR).glob('*.wav'))
    total = len(audio_files)
    
    print(f'{total} arquivos para transcrever')
    print()
    
    # Transcrever cada áudio
    resultados = []
    inicio_total = time.time()
    
    for i, filepath in enumerate(audio_files, 1):
        print(f'--- [{i}/{total}] ---')
        
        resultado = transcribe_audio(model, str(filepath), OUTPUT_DIR)
        
        if resultado:
            resultados.append(resultado)
            # Atualizar status no catálogo DuckDB
            update_catalog_status(DB_PATH, resultado['arquivo'], 'concluido')
        else:
            basename = filepath.stem
            update_catalog_status(DB_PATH, basename, 'erro')
        
        print()
    
    # Resumo final
    tempo_total = time.time() - inicio_total
    
    print('='*60)
    print('        RESUMO DA TRANSCRIÇÃO        ')
    print('='*60)
    print(f'   Arquivos processados: {len(resultados)}/{total}')
    print(f'   Total de segmentos: {sum(r["segmentos"] for r in resultados)}')
    print(f'   Tempo total: {format_time(tempo_total)}')
    print(f'   Fim: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('='*60)
    
    # Salvar log em JSON para referência futura
    log = {
        'modelo': MODEL_SIZE,
        'device': 'cpu',
        'compute_type': 'int8',
        'inicio': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tempo_total_segundos': round(tempo_total, 1),
        'resultados': resultados,
    }
    
    log_path = f'{OUTPUT_DIR}/transcricao_log.json'
    with open(log_path, 'w', encoding='utf-8') as f:
        # json.dump() escreve o dicionário como JSON no arquivo
        # ensure_ascii=False: preserva acentos (ã, é, ç)
        # indent=2: formata com indentação para legibilidade
        json.dump(log, f, ensure_ascii=False, indent=2)
    
    print(f'\nLog salvo em: {log_path}')