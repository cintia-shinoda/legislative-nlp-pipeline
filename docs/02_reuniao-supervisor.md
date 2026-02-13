# Reunião com Supervisor — 13/02/2026
## Semana 2: Resultados da Análise

### Status do Projeto

| Fase | Atividade | Status |
|------|-----------|--------|
| 1. Setup | Python 3.11, venv, dependências | OK |
| 2. Estudo | Modelos testados, fontes mapeadas | OK |
| 3. Coleta | 6 sessões, 12h28min, 8 GB WAV | OK |
| 4. Transcrição | faster-whisper large-v3, 11.435 segmentos | OK |
| 5. Pré-processamento | spaCy pt_core_news_lg, 9.239 segmentos limpos | OK |
| 6. Sentimento | cardiffnlp/xlm-roberta, 71% confiança | OK |
| 7. Tópicos | BERTopic, 29 tópicos identificados | OK |
| 8. Dashboard | Streamlit com 7 visualizações | OK |


---

### Decisões Técnicas Tomadas

| Decisão | Alternativas Testadas | Escolha | Motivo |
|---------|----------------------|---------|--------|
| Modelo transcrição | large-v3 vs medium | large-v3 | Melhor segmentação e vocabulário parlamentar |
| Modelo sentimento | nlptown vs lxyuan vs cardiffnlp | cardiffnlp/xlm-roberta | 5/6 acertos, melhor em neutros (crucial para sessões) |
| Min topic size | 10 (174 tópicos) vs 40 (29 tópicos) | 40 | Visualizável no dashboard, sem fragmentação |
| Texto para sentimento | Limpo vs original | Original | Modelo treinado em texto natural |
| Texto para embeddings | Limpo vs original | Original | sentence-transformers espera texto natural |

---

### Resultados Principais

#### Sentimento por Sessão
| Sessão | Positivo | Neutro | Negativo |
|--------|-----------|---------|-----------|
| 22/10/2025 | 8% | 82% | 9% |
| 04/11/2025 | 7% | 69% | 23% |
| 05/11/2025 | 14% | 70% | 16% |
| 11/11/2025 | 11% | 78% | 12% |
| 12/11/2025 | 11% | 76% | 13% |
| 25/11/2025 | 10% | 65% | 25% |

**Insight:** Sessões 04/11 e 25/11 significativamente mais negativas.

#### Tópicos Mais Polarizados
| Tópico | Negativo | Contexto |
|--------|-----------|----------|
| Bolsonaro / ameaça | 72% | Debate político-partidário |
| Polícia / segurança | 35% | Discussões sobre violência |
| Brasil / nacional | 28% | Temas federais polarizados |
| Proteção animal | 24% | Denúncias de maus-tratos |

#### Tópicos Mais Construtivos
| Tópico | Positivo | Contexto |
|--------|-----------|----------|
| Projetos de lei | 25% | Proposições legislativas |
| Igreja / religião | 26% | Homenagens e referências |
| Cultura / comunidades | 13% | Atividades culturais |

---

### Próximos Passos (Semana 3)

1. **Refinamento do dashboard** — melhorar UX, adicionar mais filtros
2. **Documentação técnica** — README, docstrings, relatório final
3. **Análise qualitativa** — validar amostras manualmente

---

### Pontos para Discussão

1. Os 29 tópicos são suficientes ou devemos explorar granularidade diferente?
2. O tópico #0 (catch-all, 59% dos segmentos) — vale investir em reduzi-lo?
3. Prioridade semana 3: mais sessões ou refinar análise existente?