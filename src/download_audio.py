"""
download_audio.py
Faz o download do áudio de vídeos do YouTube e salva em formato WAV.
Para executar: python src/download_audio.py
"""

# subprocess: biblioteca padrão do Python para executar comandos do terminal
# Usamos para chamar o yt-dlp como se estivéssemos digitando no terminal
import subprocess

# pathlib.Path: forma moderna do Python para lidar com caminhos de arquivos
# Melhor que concatenar strings com '/' manualmente
from pathlib import Path

# datetime: para registrar quando cada download aconteceu
from datetime import datetime


def download_audio(url, output_dir='data/raw/audio'):
    """
    Baixa apenas o áudio de um vídeo do YouTube em formato WAV.
    
    Parâmetros:
        url (str): URL do vídeo do YouTube
        output_dir (str): pasta onde salvar o áudio (padrão: data/raw/audio)
    
    Retorna:
        bool: True se o download foi bem-sucedido, False se falhou
    """
    
    # Criar a pasta de destino se não existir
    # parents=True: cria pastas intermediárias (data/ e data/raw/) se necessário
    # exist_ok=True: não dá erro se a pasta já existir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Montar o comando yt-dlp como uma lista de argumentos
    # Cada elemento da lista é um "pedaço" do comando que seria digitado no terminal
    cmd = [
        'yt-dlp',                           # O programa que vamos executar
        '-x',                               # --extract-audio: baixa só o áudio, descarta o vídeo
        '--audio-format', 'wav',            # Converte para WAV (formato sem compressão, ideal para Whisper)
        '--audio-quality', '0',             # Melhor qualidade possível (0 = máxima)
        '--no-playlist',                    # Se a URL for de playlist, baixa só o vídeo específico
        '--restrict-filenames',             # Remove caracteres especiais do nome do arquivo (acentos, |, etc.)
        '-o', f'{output_dir}/%(title)s.%(ext)s',  # Template do nome do arquivo de saída
                                            # %(title)s = título do vídeo
                                            # %(ext)s = extensão (wav)
        url                                 # A URL do vídeo a baixar
    ]
    
    # Executar o comando
    # subprocess.run() executa o comando e espera ele terminar
    # check=True: se o yt-dlp falhar, levanta uma exceção (erro) no Python
    try:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Baixando: {url}')
        subprocess.run(cmd, check=True)
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Concluído: {url}')
        return True
    except subprocess.CalledProcessError as e:
        # CalledProcessError: o yt-dlp retornou um código de erro
        # Isso pode acontecer se a URL for inválida, o vídeo foi removido, etc.
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Erro ao baixar {url}: {e}')
        return False


# Lista dos 6 vídeos selecionados para o MVP
# Câmara Municipal de São Paulo - Sessões Plenárias 2025
VIDEOS_MVP = [
    'https://www.youtube.com/watch?v=LYMmGiMGHhQ',  # 25/11/2025 ~2:05
    'https://www.youtube.com/watch?v=qJ8c3rSecRM',  # 12/11/2025 ~2:54
    'https://www.youtube.com/watch?v=aMsbmDaGtAo',  # 11/11/2025 ~1:42
    'https://www.youtube.com/watch?v=SFToio4Rz2E',  # 05/11/2025 ~1:46
    'https://www.youtube.com/watch?v=Nq4wkZODLyY',  # 04/11/2025 ~1:45
    'https://www.youtube.com/watch?v=ui3XmWQ2XUU',  # 22/10/2025 ~2:12
]


# Este bloco só executa quando rodamos o arquivo diretamente:
#   python src/download_audio.py
# Não executa se outro script importar este módulo:
#   from src.download_audio import download_audio
if __name__ == '__main__':
    print('='*60)
    print('Download de Áudio — Câmara Municipal de São Paulo')
    print(f'   Total de vídeos: {len(VIDEOS_MVP)}')
    print('='*60)
    
    # Contadores para o resumo final
    sucesso = 0
    falha = 0
    
    # Iterar sobre cada URL e baixar
    # enumerate() adiciona um índice (i) começando de 1
    for i, url in enumerate(VIDEOS_MVP, 1):
        print(f'\n--- Vídeo {i}/{len(VIDEOS_MVP)} ---')
        
        if download_audio(url):
            sucesso += 1
        else:
            falha += 1
    
    # Resumo final
    print('\n' + '='*60)
    print(f'Resumo: {sucesso} sucesso | {falha} falha')
    print('='*60)