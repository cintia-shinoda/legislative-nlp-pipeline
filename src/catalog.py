"""
catalog.py
Cria e gerencia o catálogo de sessões no DuckDB.
Armazena metadados de cada sessão: data, duração, tamanho, status do pipeline.
Para executar: python src/catalog.py
"""

# duckdb: BD SQL analítico. Roda localmente e suporta Parquet
import duckdb

# pathlib.Path: para manipular caminhos de arquivos
from pathlib import Path

# os: para obter informações do sistema (tamanho de arquivos)
import os

# mutagen.mp4 e wave: para extrair duração de arquivos de áudio
# wave é da biblioteca padrão do Python
import wave


def get_wav_duration(filepath):
    """
    Calcula a duração de um arquivo WAV em minutos.
    
    Como funciona:
    - Abre o arquivo WAV e lê os metadados do cabeçalho
    - frames = número total de amostras de áudio
    - framerate = amostras por segundo (ex: 44100 Hz)
    - duração = frames / framerate (em segundos)
    - converte para minutos dividindo por 60
    
    Parâmetros:
        filepath (str): caminho para o arquivo WAV
    
    Retorna:
        float: duração em minutos (ex: 105.5 = 1h45min30s)
    """
    try:
        with wave.open(str(filepath), 'r') as wav_file:
            frames = wav_file.getnframes()       # Total de frames (amostras)
            framerate = wav_file.getframerate()   # Amostras por segundo (Hz)
            duration_sec = frames / framerate     # Duração em segundos
            return round(duration_sec / 60, 1)    # Converter para minutos, 1 casa decimal
    except Exception as e:
        print(f'Erro ao ler {filepath}: {e}')
        return 0.0


def get_file_size_mb(filepath):
    """
    Retorna o tamanho de um arquivo em MB.
    
    Parâmetros:
        filepath (str): caminho para o arquivo
    
    Retorna:
        float: tamanho em MB (ex: 1100.5 = ~1.1 GB)
    """
    # os.path.getsize() retorna o tamanho em bytes
    # Dividimos por 1024*1024 para converter para MB
    size_bytes = os.path.getsize(filepath)
    return round(size_bytes / (1024 * 1024), 1)


def extract_date_from_filename(filename):
    """
    Extrai a data da sessão a partir do nome do arquivo.

    O nome segue o padrão: Sessao_Plenaria_DD_MM_YYYY.wav
    Precisamos reorganizar para YYYY-MM-DD (formato SQL padrão)
    
    Parâmetros:
        filename (str): nome do arquivo (sem o caminho)
    
    Retorna:
        str: data no formato 'YYYY-MM-DD' ou 'desconhecida' se falhar
    """
    try:
        # Remover extensão (.wav) e prefixo (Sessao_Plenaria_)
        # 'Sessao_Plenaria_04_11_2025.wav' -> '04_11_2025'
        name = filename.replace('.wav', '')
        parts = name.split('_')
        
        # parts = ['Sessao', 'Plenaria', '04', '11', '2025']
        # Os 3 últimos elementos são dia, mês, ano
        day = parts[-3]    # '04'
        month = parts[-2]  # '11'
        year = parts[-1]   # '2025'
        
        return f'{year}-{month}-{day}'  # '2025-11-04'
    except (IndexError, ValueError):
        return 'desconhecida'


def create_catalog(db_path='data/catalogo.duckdb', audio_dir='data/raw/audio'):
    """
    Cria o banco de dados DuckDB e popula com metadados dos áudios.
    
    Passos:
    1. Conecta (ou cria) o banco de dados
    2. Cria a tabela 'sessoes' se não existir
    3. Escaneia a pasta de áudios
    4. Para cada arquivo WAV, extrai metadados e insere na tabela
    
    Parâmetros:
        db_path (str): caminho para o arquivo do banco de dados
        audio_dir (str): pasta onde estão os áudios WAV
    """
    
    # Criar pasta para o banco se não existir
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Conectar ao DuckDB
    # Se o arquivo não existir, o DuckDB cria automaticamente
    # É um banco de dados local
    con = duckdb.connect(db_path)
    
    # Criar a tabela 'sessoes'
    # IF NOT EXISTS: só cria se a tabela ainda não existir (idempotente)
    # Cada coluna tem um tipo definido:
    #   VARCHAR = texto, DATE = data, INTEGER = número inteiro, FLOAT = decimal
    con.execute('''
        CREATE TABLE IF NOT EXISTS sessoes (
            id INTEGER PRIMARY KEY,
            titulo VARCHAR,
            data_sessao DATE,
            camara VARCHAR DEFAULT 'municipal_sp',
            duracao_min FLOAT,
            tamanho_mb FLOAT,
            arquivo_audio VARCHAR,
            status_transcricao VARCHAR DEFAULT 'pendente',
            status_sentimento VARCHAR DEFAULT 'pendente',
            status_topicos VARCHAR DEFAULT 'pendente'
        )
    ''')
    
    # Limpar tabela antes de re-popular (para evitar duplicatas)
    # DELETE FROM remove todas as linhas, mas mantém a estrutura
    con.execute('DELETE FROM sessoes')
    
    # Escanear pasta de áudios
    # Path(audio_dir).glob('*.wav') retorna todos os arquivos que terminam em .wav
    # sorted() ordena alfabeticamente (que neste caso = ordem cronológica)
    audio_files = sorted(Path(audio_dir).glob('*.wav'))
    
    print(f'Encontrados {len(audio_files)} arquivos WAV em {audio_dir}/')
    print()
    
    for i, filepath in enumerate(audio_files, 1):
        # filepath.name = só o nome do arquivo, sem o caminho
        # Ex: 'Sessao_Plenaria_04_11_2025.wav'
        filename = filepath.name
        
        # Extrair metadados
        data_sessao = extract_date_from_filename(filename)
        duracao = get_wav_duration(filepath)
        tamanho = get_file_size_mb(filepath)
        
        # Formatar duração para exibição
        horas = int(duracao // 60)
        minutos = int(duracao % 60)
        
        print(f'  {i}. {filename}')
        print(f'     Data: {data_sessao} | Duração: {horas}h{minutos:02d}min | Tamanho: {tamanho:.0f} MB')
        
        # Inserir no banco de dados
        # Os ? são placeholders — o DuckDB substitui pelos valores da tupla
        # Previne SQL injection
        con.execute('''
            INSERT INTO sessoes (id, titulo, data_sessao, duracao_min, tamanho_mb, arquivo_audio)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', [i, filename.replace('.wav', ''), data_sessao, duracao, tamanho, str(filepath)])
    
    # Mostrar resumo do banco
    print()
    print('='*60)
    print('Catálogo criado com sucesso!')
    print('='*60)
    
    # Executar uma consulta SQL para mostrar o resumo
    # .fetchdf() converte o resultado para um DataFrame do Pandas (tabela)
    resultado = con.execute('''
        SELECT 
            COUNT(*) as total_sessoes,
            ROUND(SUM(duracao_min), 0) as minutos_totais,
            ROUND(SUM(tamanho_mb) / 1024, 1) as gb_totais
        FROM sessoes
    ''').fetchdf()
    
    total = resultado.iloc[0]  # Primeira (e única) linha do resultado
    print(f'   Sessões: {int(total["total_sessoes"])}')
    print(f'   Duração total: {int(total["minutos_totais"])} min (~{int(total["minutos_totais"])//60}h{int(total["minutos_totais"])%60:02d})')
    print(f'   Tamanho total: {total["gb_totais"]} GB')
    
    # Fechar conexão
    con.close()


# Bloco de execução direta
if __name__ == '__main__':
    create_catalog()