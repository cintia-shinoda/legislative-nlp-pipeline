"""
preprocess.py
Limpa e pré-processa as transcrições
Para executar: python src/preprocess.py
"""

import spacy
import pandas as pd
import duckdb
from pathlib import Path
import re
import time


def load_nlp():
    """
    Carrega o modelo spaCy de português.
    
    Retorna:
        nlp: modelo spaCy carregado
    """
    print('Carregando spaCy pt_core_news_lg...')
    nlp = spacy.load('pt_core_news_lg')
    print('Carregado!')
    return nlp


def is_vinheta(text):
    """
    Detecta se um segmento é vinheta/propaganda da Rede Câmara.
    
    Usa expressões regulares (regex) para identificar padrões comuns
    nas vinhetas que aparecem no início e fim dos vídeos.
    
    Parâmetros:
        text (str): texto do segmento
    
    Retorna:
        bool: True se for vinheta, False se for conteúdo real
    """
    # Lista de padrões que indicam vinheta
    # re.IGNORECASE = não diferencia maiúsculas de minúsculas
    # O '|' dentro do regex significa 'OU'
    padroes_vinheta = [
        r'rede câmara',                    # Menção à Rede Câmara
        r'sua conexão com a política',      # Slogan da Rede Câmara
        r'siga.*instagram',                 # Promoção de redes sociais
        r'arroba.*câmara',                  # Menção ao @ de redes sociais
        r'inscreva-se',                     # Call-to-action do YouTube
        r'histórias da câmara',             # Nome de programa
        r'diversidade fica por aqui',       # Encerramento de programa
        r'tchau.*tchau',                    # Despedida de programa
    ]
    
    # any() retorna True se QUALQUER padrão for encontrado
    # re.search() procura o padrão em qualquer posição do texto
    return any(
        re.search(padrao, text, re.IGNORECASE) 
        for padrao in padroes_vinheta
    )


def is_segment_useful(text, min_words=3):
    """
    Verifica se um segmento tem conteúdo útil para análise.
    
    Filtra segmentos muito curtos ou sem valor analítico.
    
    Parâmetros:
        text (str): texto do segmento
        min_words (int): número mínimo de palavras para ser útil
    
    Retorna:
        bool: True se o segmento for útil
    """
    # Contar palavras
    # split() sem argumentos divide por qualquer espaço em branco
    words = text.strip().split()
    
    if len(words) < min_words:
        return False
    
    # Padrões de saudação/despedida sem valor analítico
    saudacoes = [
        r'^boa (tarde|noite|dia)',
        r'^muito obrigad[oa]',
        r'^obrigad[oa]',
        r'^com licença',
    ]
    
    # ^ no regex significa "início da string"
    return not any(
        re.search(padrao, text.strip(), re.IGNORECASE) 
        for padrao in saudacoes
    )


def extract_entities(doc):
    """
    Extrai entidades nomeadas de um documento spaCy.
    
    Parâmetros:
        doc: documento spaCy processado
    
    Retorna:
        dict: dicionário com listas de entidades por tipo
              Ex: {'PER': ['Cris Monteiro'], 'ORG': ['Câmara Municipal']}
    """
    entities = {}
    for ent in doc.ents:
        # ent.label_ = tipo (PER, LOC, ORG, MISC)
        # ent.text = texto da entidade
        if ent.label_ not in entities:
            entities[ent.label_] = []
        # Evitar duplicatas dentro do mesmo segmento
        if ent.text not in entities[ent.label_]:
            entities[ent.label_].append(ent.text)
    return entities


def clean_text(doc):
    """
    Limpa o texto removendo stopwords e pontuação, mas PRESERVANDO entidades.
    
    Este é o fix para o problema 'São Paulo' -> 'paulo':
    primeiro identifica quais tokens fazem parte de entidades,
    depois só remove stopwords dos tokens que NÃO são entidades.
    
    Parâmetros:
        doc: documento spaCy processado
    
    Retorna:
        str: texto limpo com lemas
    """
    # Conjunto de tokens que fazem parte de entidades
    # Usamos um set para busca rápida (O(1) em vez de O(n))
    entity_tokens = set()
    for ent in doc.ents:
        # ent é um Span (grupo de tokens). Iteramos sobre cada token dentro dele
        for token in ent:
            entity_tokens.add(token.i)  # token.i = índice do token no documento
    
    tokens_limpos = []
    for token in doc:
        # Se o token faz parte de uma entidade, manter como está
        if token.i in entity_tokens:
            tokens_limpos.append(token.text.lower())
        # Se NÃO é entidade: aplicar filtros de limpeza
        elif (not token.is_stop          # Não é stopword
              and not token.is_punct      # Não é pontuação
              and len(token.text) > 1     # Tem mais de 1 caractere
              and token.pos_ != 'NUM'):   # Não é número
            tokens_limpos.append(token.lemma_.lower())
    
    return ' '.join(tokens_limpos)


def preprocess_session(nlp, parquet_path, output_dir='data/processed'):
    """
    Pré-processa uma sessão inteira: filtra, limpa, extrai entidades.
    
    Parâmetros:
        nlp: modelo spaCy carregado
        parquet_path (str): caminho do Parquet com transcrição bruta
        output_dir (str): pasta para salvar resultado
    
    Retorna:
        dict: estatísticas do processamento
    """
    basename = Path(parquet_path).stem
    print(f'Processando: {basename}')
    
    # Ler transcrição bruta
    df = pd.read_parquet(parquet_path)
    original_count = len(df)
    
    # PASSO 1: Filtrar vinhetas
    # ~ inverte o booleano: True vira False e vice-versa
    # Ou seja: manter linhas onde is_vinheta() retorna FALSE
    df['is_vinheta'] = df['text'].apply(is_vinheta)
    df_filtered = df[~df['is_vinheta']].copy()
    vinhetas_removed = original_count - len(df_filtered)
    
    # PASSO 2: Filtrar segmentos inúteis
    df_filtered = df_filtered[
        df_filtered['text'].apply(is_segment_useful)
    ].copy()
    useless_removed = original_count - vinhetas_removed - len(df_filtered)
    
    # PASSO 3: Processar com spaCy
    # nlp.pipe() processa múltiplos textos em batch (mais eficiente que um por um)
    # batch_size=50: processa 50 textos de cada vez na memória
    texts = df_filtered['text'].tolist()
    
    clean_texts = []
    all_entities = []
    
    for doc in nlp.pipe(texts, batch_size=50):
        clean_texts.append(clean_text(doc))
        all_entities.append(extract_entities(doc))
    
    # Adicionar colunas ao DataFrame
    df_filtered['text_clean'] = clean_texts
    df_filtered['entities'] = all_entities
    
    # PASSO 4: Remover segmentos que ficaram vazios após limpeza
    df_filtered = df_filtered[
        df_filtered['text_clean'].str.strip().str.len() > 0
    ].copy()
    
    # Remover coluna auxiliar
    # axis=1 indica que estamos removendo uma COLUNA (não uma linha)
    df_filtered = df_filtered.drop(columns=['is_vinheta'])
    
    # Resetar índice
    # drop=True: descarta o índice antigo em vez de criar uma coluna com ele
    df_filtered = df_filtered.reset_index(drop=True)
    
    # Salvar resultado
    output_path = f'{output_dir}/{basename}_clean.parquet'
    df_filtered.to_parquet(output_path, index=False)
    
    # Estatísticas
    stats = {
        'arquivo': basename,
        'segmentos_original': original_count,
        'vinhetas_removidas': vinhetas_removed,
        'inuteis_removidos': useless_removed,
        'segmentos_final': len(df_filtered),
        'palavras_original': df['text'].str.split().str.len().sum(),
        'palavras_clean': df_filtered['text_clean'].str.split().str.len().sum(),
    }
    
    print(f'{original_count} → {len(df_filtered)} segmentos '
          f'(-{vinhetas_removed} vinhetas, -{useless_removed} curtos/saudações) '
          f'| Salvo: {output_path}')
    
    return stats


# ============================================================
#                   EXECUÇÃO PRINCIPAL
# ============================================================
if __name__ == '__main__':
    PROCESSED_DIR = 'data/processed'
    
    print('='*60)
    print('Pré-processamento — spaCy pt_core_news_lg')
    print('='*60)
    print()
    
    # Carregar modelo
    nlp = load_nlp()
    print()
    
    # Encontrar transcrições brutas (Parquets SEM '_clean' no nome)
    # O filtro evita reprocessar arquivos já limpos
    parquets = sorted([
        p for p in Path(PROCESSED_DIR).glob('*.parquet')
        if '_clean' not in p.stem
        and 'transcricao_log' not in p.stem
    ])
    
    print(f'{len(parquets)} transcrições para processar')
    print()
    
    all_stats = []
    start = time.time()
    
    for i, pq in enumerate(parquets, 1):
        print(f'--- [{i}/{len(parquets)}] ---')
        stats = preprocess_session(nlp, str(pq), PROCESSED_DIR)
        all_stats.append(stats)
        print()
    
    elapsed = time.time() - start
    
    # Resumo final
    print('='*60)
    print('        RESUMO DO PRÉ-PROCESSAMENTO        ')
    print('='*60)
    
    total_orig = sum(s['segmentos_original'] for s in all_stats)
    total_final = sum(s['segmentos_final'] for s in all_stats)
    total_vinhe = sum(s['vinhetas_removidas'] for s in all_stats)
    total_words_orig = sum(s['palavras_original'] for s in all_stats)
    total_words_clean = sum(s['palavras_clean'] for s in all_stats)
    
    print(f'   Segmentos: {total_orig:,} → {total_final:,} ({total_final/total_orig:.0%} mantidos)')
    print(f'   Vinhetas removidas: {total_vinhe}')
    print(f'   Palavras: {total_words_orig:,} (bruto) → {total_words_clean:,} (limpo)')
    print(f'   Tempo total: {elapsed:.0f}s ({elapsed/60:.1f} min)')
    print('='*60)