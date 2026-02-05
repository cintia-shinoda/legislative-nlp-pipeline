# Reunião 1 — Supervisor

## Status do Projeto

### Concluído (Dias 1-4)
- Ambiente Python 3.11 + venv configurado
- Dependências instaladas e testadas
- Estrutura do projeto no Git
- 6 sessões plenárias baixadas (Câmara Municipal SP)
  - 12h28min de áudio total | 8.0 GB
  - Período: out-nov 2025
- Catálogo de metadados no DuckDB

### Próximos Passos (Dias 5-7)
- Transcrever os 6 áudios com faster-whisper (large-v3)
- Avaliar qualidade das transcrições (WER manual)
- Indexar transcrições no DuckDB

## Pontos para Discussão
- Validar a escolha da Câmara Municipal SP como fonte principal
  (Câmara dos Deputados não tem playlists acessíveis)
- Quantas sessões são suficientes para o MVP?
- Expectativas para a apresentação ao stakeholder
