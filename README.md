# ML Retenção Forecast — Projeção de Vendas, Retiradas e Crescimento (2027)

Projeto de Machine Learning e MLOps construído a partir dos dados do Data Warehouse do **`dash-retencao`** (The Fiber), utilizando modelos da família de **Regressão Linear** (OLS, Ridge, Lasso, ElasticNet e Huber) com ponderação operacional de **dias úteis** para projetar **Vendas (Ativações)**, **Retiradas (Cancelamentos totais do mês, independente da safra)** e o **Crescimento da Carteira (Planta)** ao longo de **2027**.

---

## 1. Regras de Negócio Incorporadas

- **Fonte de dados**: Leitura direta (somente leitura) do PostgreSQL DW (`fato_churn_contrato` + tabelas De-Para `dashboard_funcionarioixc`, `dashboard_canalvenda`, `dashboard_bairrorede`, `dashboard_motivocancelamentoixc`), replicando integralmente as regras de limpeza do `dash-retencao` (desduplicação por `id_contrato`, `ignorar_metricas`, `mes_transferencia`, exclusão de `Vendedor Padrão` nas ativações e normalização de cidades/redes).
- **Retirada**: Todo contrato cancelado (`flag_churn = 1`, motivos válidos) contabilizado no mês do cancelamento (`anomes_cancelamento`), independente da safra de ativação.
- **Calendário de Dias Úteis Ponderados (2024–2027)**:
  - **Segunda a Sexta (não feriado)**: peso `1.0`
  - **Sábado (não feriado)**: peso `0.5`
  - **Domingo ou Feriado** (nacionais, estaduais GO, municipais Goiânia e Carnaval/Corpus Christi): peso `0.0`
- **Tratamento estatístico de anomalias e censura**:
  - Exclusão automática do mês corrente ainda aberto (incompleto).
  - Sinalização de censura à direita no `FPD 61d` para os últimos meses sem maturação completa.
  - Detecção transparente de picos de baixa em lote de cancelamentos (limpeza de base IXC em `2025-06` e `2025-08`).

---

## 2. Estrutura MLOps do Projeto

```text
ml-retencao-forecast/
├── configs/
│   └── params.yaml                  # Pesos do calendário, feriados, hiperparâmetros e cenários 2027
├── data/
│   ├── raw/                         # Snapshots extraídos do PostgreSQL DW (Parquet + metadata.json)
│   ├── processed/                   # Séries mensais/diárias tratadas, calendário, bairros e canais
│   └── predictions/                 # Projeções mensais 2027 (CSV, Parquet e resumo_2027.json)
├── models/                          # Modelos treinados (.joblib), benchmarks (.csv) e metadata.json
├── notebooks/                       # Documentação padrão CRISP-DM executada com gráficos e tabelas
│   ├── 01_analise_exploratoria_eda.ipynb
│   ├── 02_engenharia_de_atributos.ipynb
│   ├── 03_modelagem_regressao_linear.ipynb
│   └── 04_projecao_2027_e_cenarios.ipynb
├── reports/
│   └── figures/                     # Gráficos exportados em PNG de alta resolução
├── src/
│   ├── config.py                    # Carregamento de params.yaml e credenciais read-only
│   ├── pipeline.py                  # Orquestrador CLI (extract -> features -> train -> predict)
│   ├── data/extract.py              # Extrator do DW com regras de De-Para
│   ├── features/calendar.py         # Motor de calendário ponderado 2024–2027 + validação empírica
│   ├── features/build.py            # Engenharia de atributos, lags, anomalias, bairros e canais
│   └── models/
│       ├── train.py                 # Benchmark Regressão Linear + TimeSeriesSplit + holdout 2026
│       └── predict.py               # Simulador recursivo em cascata para 2027 (3 cenários)
├── tests/
│   └── test_pipeline.py             # Testes automatizados de calendário, integridade e projeção
├── requirements.txt
└── .env.example
```

---

## 3. Como Executar o Pipeline Completo

```bash
cd /home/desenv/projetos/ml-retencao-forecast

# 1. Ativar ambiente virtual
source .venv/bin/activate

# 2. Executar pipeline MLOps completo (DW -> Features -> Treino -> Projeção 2027)
python -m src.pipeline --step all

# Ou executar etapas individuais:
python -m src.pipeline --step extract
python -m src.pipeline --step features
python -m src.pipeline --step train
python -m src.pipeline --step predict

# 3. Rodar testes automatizados
pytest -v

# 4. Abrir os notebooks no JupyterLab
jupyter lab notebooks/
```

---

## 4. Resumo dos Resultados (Execução Atual)

- **Campeão Vendas**: `ElasticNet_Taxa_DU` (modelando `vendas / dias_uteis_ponderados`) — Holdout 2026: **MAPE 2,55%**, **MAE 50,9 vendas/mês**.
- **Campeão Retiradas**: `ElasticNet_Direto_Limpo` (tratando anomalias de baixa em lote) — Holdout 2026: **MAPE 3,42%**, **MAE 33,3 retiradas/mês**, **R² 0,42**.
- **Projeção 2027 (273,5 dias úteis ponderados no ano — partindo de 35.645 clientes em Jan/2027)**:

| Cenário 2027 | Vendas 2027 | Retiradas 2027 | Saldo Líquido | Carteira Fim 2027 | Crescimento 2027 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Conservador** | 21.377 | 13.977 | +7.400 | 43.045 | +20,8% |
| **Base** | 24.430 | 13.917 | +10.513 | 46.157 | +29,5% |
| **Expansão** (+bairros e +1 canal) | 31.106 | 15.008 | +16.098 | 51.742 | +45,2% |
