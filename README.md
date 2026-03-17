# Análise de Sentimento em Discursos Legislativos

Pipeline de análise automatizada de sentimento e tópicos em sessões plenárias da Câmara Municipal de São Paulo, utilizando técnicas de Processamento de Linguagem Natural (NLP - *Natural Language Processing*) e Aprendizado de Máquina.


![badge](https://img.shields.io/badge/status-in%20progress-yellow)
[![GitHub forks](https://img.shields.io/github/forks/Naereen/StrapDown.js.svg?style=social&label=Fork&maxAge=2592000)](https://github.com/cintia-shinoda/legislative-nlp-pipeline) [![GitHub stars](https://img.shields.io/github/stars/Naereen/StrapDown.js.svg?style=social&label=Star&maxAge=2592000)](https://github.com/cintia-shinoda/legislative-nlp-pipeline/stargazers/)


---

## Arquitetura

```bash
legislative-nlp-pipeline
├── data/
│   ├── output/
│   ├── processed/
│   ├── raw/
│   └── catalogo.duckdb
│
├── docs/
│   └── fontes.md
│
├── notebooks/
│
├── src/
│   ├── catalog.py
│   ├── dashboard.py
│   ├── download_audio.py
│   ├── preprocess.py
│   ├── sentiment.py
│   ├── test_pipeline
│   ├── topics.py
│   └── transcribe.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Stack Tecnológico
- **Linguagem de Programação**: Python
- **Frameworks e Bibliotecas**: yt-dlp,Faster Whisper, spaCy, Pandas, Plotly, Transformers, tiktoken, Streamlit, WordCloud
- **Banco de Dados**: DuckDB

---

## Pipeline
```bash
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   yt-dlp    │───▶│faster-whisper│───▶│   spaCy      │───▶│  BERTimbau   │
│  (download) │    │ (transcrição)│    │  (limpeza)   │    │ (sentimento) │
│             │    │              │    │              │    │              │
│ URL → .wav  │    │ .wav → texto │    │ texto bruto  │    │ texto limpo  │
│             │    │              │    │  → limpo     │    │  → score     │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                  │
                                            ┌─────────────────────┘
                                            ▼
                                     ┌──────────────┐    ┌──────────────┐
                                     │  BERTopic    │───▶│  Streamlit   │
                                     │  (tópicos)   │    │ (dashboard)  │
                                     │              │    │              │
                                     │ textos →     │    │ dados →      │
                                     │  clusters    │    │  gráficos    │
                                     └──────────────┘    └──────────────┘
```

---

## Resultados do MVP

<img src="images/1.png"> 
<img src="images/2.png">
<img src="images/3.png">
<img src="images/4.png">
<img src="images/5.png">
<img src="images/6.png">
<img src="images/7.png">

---

## Como executar o projeto
1. Clone o repositório:
```bash
git clone https://github.com/cintia-shinoda/legislative-nlp-pipeline.git
```

2. Entre na pasta do projeto:
```bash
cd legislative-nlp-pipeline
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```