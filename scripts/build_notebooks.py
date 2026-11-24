"""
Gera os 4 Jupyter Notebooks padronizados (CRISP-DM / MLOps) em `notebooks/`.
"""
from __future__ import annotations

from pathlib import Path
import nbformat as nbf

ROOT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"


def _make_nb(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (.venv)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.10"}
    nb.cells = [
        nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text)
        for kind, text in cells
    ]
    return nb


def build_nb_01() -> nbf.NotebookNode:
    cells = [
        ("md", """# 01. Análise Exploratória de Dados (EDA) — Vendas, Retiradas, FPD, Dias Úteis, Bairros e Canais

**Etapa CRISP-DM:** Entendimento do Negócio e Entendimento dos Dados  
**Fonte:** Data Warehouse `db_thefiber` (`fato_churn_contrato` + tabelas De-Para do `dash-retencao`)

### Objetivos deste notebook
1. Auditar a qualidade dos dados extraídos do DW e validar a separação entre **meses fechados** e **mês parcial corrente**.
2. Analisar a evolução histórica (2024–2026) de **Vendas (Ativações)**, **Retiradas (Cancelamentos totais do mês, independente da safra)**, **Saldo Líquido**, **Carteira (Planta)** e **FPD (61 dias)** — incluindo a censura à direita dos meses recentes.
3. Investigar **anomalias de retiradas** (baixas em lote no IXC) que poderiam distorcer uma Regressão Linear.
4. Validar empiricamente a regra de negócio de **Dias Úteis Ponderados (`1,0` útil / `0,5` sábado / `0,0` domingo e feriado)** usando a série diária real.
5. Realizar análise exploratória estratégica sobre **Novos Bairros (curva de ramp-up por Tipo de Rede)** e **Novos Canais de Venda** para embasar as variáveis do modelo e os cenários de 2027."""),
        ("code", """import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path("..").resolve()
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (13, 5)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11

with open(DATA_RAW / "extract_metadata.json", encoding="utf-8") as f:
    ext_meta = json.load(f)
with open(DATA_PROC / "features_metadata.json", encoding="utf-8") as f:
    feat_meta = json.load(f)

df_mensal = pd.read_parquet(DATA_PROC / "serie_mensal_ml.parquet")
df_closed = df_mensal[df_mensal["mes_fechado"]].copy()
df_diaria = pd.read_parquet(DATA_PROC / "serie_diaria.parquet")
df_pesos = pd.read_parquet(DATA_PROC / "pesos_empiricos_dias.parquet")
df_bairros = pd.read_parquet(DATA_PROC / "analise_bairros.parquet")
df_canais = pd.read_parquet(DATA_PROC / "analise_canais.parquet")

print("=== AUDITORIA DA EXTRAÇÃO (DW -> PARQUET) ===")
print(f"Contratos extraídos (desduplicados): {ext_meta['rows_deduplicated']:,}")
print(f"Período completo no DW: {ext_meta['min_anomes_ativacao']} até {ext_meta['max_anomes_ativacao']}")
print(f"Última data de ativação: {ext_meta['max_data_ativacao']} | Cancelamento: {ext_meta['max_data_cancelamento']}")
print(f"Último mês fechado considerado para ML: {feat_meta['last_closed_month']} ({feat_meta['closed_months_count']} meses fechados)")
print(f"Carteira base (Dez/2023): {feat_meta['carteira_inicial_2023_12']:,} -> Carteira atual ({feat_meta['last_closed_month']}): {feat_meta['carteira_ultimo_mes_fechado']:,}")
"""),
        ("md", """## 1. Série Histórica: Vendas, Retiradas, Saldo Líquido e Carteira (2024–2026)

Abaixo visualizamos a dinâmica mensal nos meses fechados, destacando automaticamente:
- A forte aceleração comercial a partir de 2025;
- Os **dois picos atípicos de retiradas (`2025-06` e `2025-08`)**, característicos de baixa em lote de inadimplência acumulada no IXC;
- A diferença entre **volume bruto de vendas** e **vendas por dia útil ponderado** (explicando imediatamente quedas de calendário como Fevereiro/2026 e picos como Julho/2026)."""),
        ("code", """resumo_anual = (
    df_closed.groupby("ano", as_index=False)
    .agg(
        meses=("anomes", "count"),
        du_ponderados=("dias_uteis_ponderados", "sum"),
        vendas=("vendas", "sum"),
        retiradas=("retiradas", "sum"),
        inadimplencia=("retiradas_inadimplencia", "sum"),
        voluntario=("retiradas_voluntario", "sum"),
        saldo_liquido=("saldo_liquido", "sum"),
        carteira_fim=("carteira_fim_mes", "last"),
        bairros_ativos_media=("qtd_bairros_ativos_mes", "mean"),
        canais_ativos_media=("qtd_canais_ativos_mes", "mean"),
    )
)
resumo_anual["vendas_por_du"] = (resumo_anual["vendas"] / resumo_anual["du_ponderados"]).round(2)
resumo_anual["retirada_sobre_venda_pct"] = (resumo_anual["retiradas"] / resumo_anual["vendas"] * 100).round(1)
display(resumo_anual.round(1))

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# Painel 1: Vendas vs Retiradas com destaque de anomalias
ax = axes[0]
ax.plot(df_closed["anomes"], df_closed["vendas"], marker="o", lw=2.3, color="#1f77b4", label="Vendas (Ativações)")
ax.plot(df_closed["anomes"], df_closed["retiradas"], marker="s", lw=2.0, color="#d62728", label="Retiradas (Cancelamentos Totais)")
anom = df_closed[df_closed["anomalia_retirada"] == 1]
ax.scatter(anom["anomes"], anom["retiradas"], s=150, color="#ff7f0e", edgecolor="black", zorder=5, label="Anomalia de Baixa em Lote (IXC)")
for _, r in anom.iterrows():
    ax.annotate(f"Anomalia {r['anomes']}\\n({r['retiradas']} ret.)", (r["anomes"], r["retiradas"]),
                textcoords="offset points", xytext=(8, 6), fontsize=9, fontweight="bold", color="#b34700")
ax.set_title("Série Histórica Mensal — Vendas vs. Retiradas (Meses Fechados)")
ax.set_ylabel("Contratos / Mês")
ax.legend(loc="upper left")

# Painel 2: Evolução da Carteira (Planta) e Saldo Líquido
ax2 = axes[1]
bars = ax2.bar(df_closed["anomes"], df_closed["saldo_liquido"],
               color=np.where(df_closed["saldo_liquido"] >= 0, "#2ca02c", "#d62728"), alpha=0.65, label="Saldo Líquido Mensal")
ax2.set_ylabel("Saldo Líquido (Vendas - Retiradas)")
ax2_r = ax2.twinx()
ax2_r.plot(df_closed["anomes"], df_closed["carteira_fim_mes"], color="#1b4f72", lw=2.5, marker="o", markersize=4, label="Carteira Ativa (Planta)")
ax2_r.set_ylabel("Carteira Ativa (Contratos)")
ax2.set_title("Crescimento da Carteira Ativa (Planta) e Saldo Líquido Mensal")
ax2.tick_params(axis="x", rotation=45)

h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2_r.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="upper left")

plt.tight_layout()
plt.savefig(FIG_DIR / "01_serie_historica_vendas_retiradas_planta.png", dpi=140)
plt.show()
"""),
        ("md", """## 2. Validação Empírica dos Pesos de Dias Úteis (`1,0` / `0,5` / `0,0`)

Comparamos a regra operacional adotada (`Dia Útil = 1,0`, `Sábado = 0,5`, `Domingo/Feriado = 0,0`) com a média empírica diária de ativações e retiradas registrada no DW de Jan/2024 a Out/2026.

Também verificamos como normalizar vendas por **dias úteis ponderados (`vendas / DU`)** elimina o ruído de calendário (ex.: Fev/2026 com apenas `20,0 DU` por causa do Carnaval vs. Jul/2026 com `25,0 DU`)."""),
        ("code", """display(df_pesos.round(3))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Peso teórico vs empírico
df_melt = df_pesos.melt(
    id_vars="tipo_dia",
    value_vars=["peso_teorico", "peso_empirico_vendas", "peso_empirico_retiradas"],
    var_name="Metrica",
    value_name="Peso",
)
df_melt["Metrica"] = df_melt["Metrica"].map({
    "peso_teorico": "Regra de Negócio (1 / 0,5 / 0)",
    "peso_empirico_vendas": "Empírico — Vendas",
    "peso_empirico_retiradas": "Empírico — Retiradas",
})
sns.barplot(data=df_melt, x="tipo_dia", y="Peso", hue="Metrica", ax=axes[0])
axes[0].set_title("Pesos por Tipo de Dia: Regra de Negócio vs. Dados Reais")
axes[0].set_xlabel("")
axes[0].set_ylabel("Peso relativo ao Dia Útil (= 1,0)")

# Gráfico 2: Efeito do calendário em 2026 (Volume bruto vs Vendas por DU)
df_2026 = df_closed[df_closed["ano"] == 2026].copy()
ax_l = axes[1]
ax_l.bar(df_2026["anomes"], df_2026["vendas"], alpha=0.55, color="#4c72b0", label="Vendas Totais (esq.)")
ax_l.set_ylabel("Vendas / Mês")
ax_l.tick_params(axis="x", rotation=45)
ax_r = ax_l.twinx()
ax_r.plot(df_2026["anomes"], df_2026["vendas_por_du"], color="#c44e52", lw=2.5, marker="o", label="Vendas / Dia Útil Ponderado (dir.)")
ax_r.set_ylabel("Vendas por Dia Útil Ponderado")
ax_l.set_title("Efeito Calendário em 2026: Fev (20 DU) vs. Jul (25 DU)")
h1, l1 = ax_l.get_legend_handles_labels()
h2, l2 = ax_r.get_legend_handles_labels()
ax_l.legend(h1 + h2, l1 + l2, loc="lower right")

plt.tight_layout()
plt.savefig(FIG_DIR / "01_validacao_dias_uteis.png", dpi=140)
plt.show()
"""),
        ("md", """## 3. Qualidade das Vendas (FPD 61 dias) e Composição das Retiradas

- **Atenção à censura à direita no FPD**: contratos ativados nos últimos ~2 a 3 meses ainda não completaram 61 dias após o vencimento da 1ª fatura; por isso o indicador cai artificialmente próximo do presente. O pipeline marca automaticamente `fpd_maturado = True` apenas até o mês de corte seguro.
- **Composição das retiradas**: separar *Inadimplência* vs. *Cancelamento Voluntário* comprova visualmente que os picos de `2025-06` e `2025-08` foram choques pontuais de baixa de inadimplência."""),
        ("code", """fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Composição das Retiradas
axes[0].stackplot(
    df_closed["anomes"],
    df_closed["retiradas_inadimplencia"],
    df_closed["retiradas_voluntario"],
    labels=["Inadimplência", "Cancelamento Voluntário"],
    colors=["#e15759", "#f28e2b"],
    alpha=0.85,
)
axes[0].set_title("Composição Mensal das Retiradas (Inadimplência vs. Voluntário)")
axes[0].set_ylabel("Contratos Cancelados")
axes[0].tick_params(axis="x", rotation=60)
axes[0].set_xticks(range(0, len(df_closed), 3))
axes[0].legend(loc="upper left")

# FPD 61d maturado vs censurado
mat = df_closed[df_closed["fpd_maturado"]]
cens = df_closed[~df_closed["fpd_maturado"]]
axes[1].plot(mat["anomes"], mat["taxa_fpd_61d"] * 100, marker="o", lw=2.2, color="#2b5c8f", label="FPD 61d Maturado (%)")
axes[1].plot(cens["anomes"], cens["taxa_fpd_61d"] * 100, marker="x", lw=1.8, ls="--", color="#999999", label="Janela Censurada (< 61d)")
axes[1].set_title("Evolução do FPD 61 Dias (Maturado vs. Censurado)")
axes[1].set_ylabel("Taxa FPD 61d (%)")
axes[1].tick_params(axis="x", rotation=60)
axes[1].set_xticks(range(0, len(df_closed), 3))
axes[1].legend(loc="upper left")

plt.tight_layout()
plt.savefig(FIG_DIR / "01_fpd_e_composicao_retiradas.png", dpi=140)
plt.show()
"""),
        ("md", """## 4. EDA Estratégica: Novos Bairros (Curva de Ramp-up) e Novos Canais de Venda

Esta seção investiga duas alavancas essenciais de crescimento para 2027:
1. **Novos Bairros**: quantas vendas e retiradas um bairro recém-aberto entrega entre o mês de estreia (`M0`) e o 6º mês (`M6`), comparando **Rede Própria** vs. **Rede Neutra**.
2. **Novos Canais de Venda**: o papel dos canais lançados entre fim de 2024 e 2026 (`Loja Lorena`, `Loja Vera Cruz`, `Loja Maysa`, `INDIQUE E GANHE`, `ORBI`) na mudança de patamar de vendas e qualidade (FPD)."""),
        ("code", """# 4.1 Curva de Ramp-up de Novos Bairros (abertos a partir de 2024-01)
bn = df_bairros[
    df_bairros["bairro_novo_periodo"]
    & df_bairros["tipo_rede_efetivo"].isin(["Rede Própria", "Rede Neutra"])
    & (df_bairros["idade_bairro_meses"] <= 6)
].copy()

ramp_up = (
    bn.groupby(["tipo_rede_efetivo", "idade_bairro_meses"], as_index=False)
    .agg(
        bairros_ativos=("bairro_norm", "nunique"),
        media_vendas=("vendas", "mean"),
        media_retiradas=("retiradas", "mean"),
    )
)
print("=== CURVA DE RAMP-UP DE NOVOS BAIRROS (M0 a M6) ===")
display(ramp_up.round(2))

# 4.2 Resumo por Canal de Venda
canais_resumo = (
    df_canais.groupby(["canal_venda_efetivo", "mes_estreia_canal"], as_index=False)
    .agg(
        meses_ativos=("anomes", "nunique"),
        vendas=("vendas", "sum"),
        retiradas=("retiradas", "sum"),
        fpd_61d=("fpd_61d", "sum"),
    )
    .sort_values("vendas", ascending=False)
)
canais_resumo["vendas_por_mes_ativo"] = (canais_resumo["vendas"] / canais_resumo["meses_ativos"]).round(1)
canais_resumo["taxa_fpd_pct"] = (canais_resumo["fpd_61d"] / canais_resumo["vendas"] * 100).round(1)
canais_resumo["retirada_sobre_venda_pct"] = (canais_resumo["retiradas"] / canais_resumo["vendas"] * 100).round(1)
print("=== DESEMPENHO POR CANAL DE VENDA (2024–2026) ===")
display(canais_resumo)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# Gráfico Ramp-up Bairros
sns.lineplot(data=ramp_up, x="idade_bairro_meses", y="media_vendas", hue="tipo_rede_efetivo", marker="o", lw=2.4, ax=axes[0])
axes[0].set_title("Curva de Ramp-up de Novos Bairros: Vendas Médias/Mês (M0–M6)")
axes[0].set_xlabel("Meses desde a estreia do bairro (M0 = mês de abertura)")
axes[0].set_ylabel("Média de Vendas por Bairro Ativo")
axes[0].legend(title="Tipo de Rede")

# Gráfico Canais (Vendas/mês ativo vs FPD%)
top_c = canais_resumo[canais_resumo["vendas"] >= 200].copy()
sns.barplot(data=top_c, y="canal_venda_efetivo", x="vendas_por_mes_ativo", hue="canal_venda_efetivo", dodge=False, legend=False, ax=axes[1])
for i, (_, r) in enumerate(top_c.iterrows()):
    axes[1].text(r["vendas_por_mes_ativo"] + 4, i, f"Estreia {r['mes_estreia_canal']} | FPD {r['taxa_fpd_pct']:.1f}%", va="center", fontsize=9)
axes[1].set_title("Ritmo de Vendas por Canal (Vendas/Mês Ativo) e Taxa FPD")
axes[1].set_xlabel("Média de Vendas por Mês Ativo")
axes[1].set_ylabel("")
axes[1].set_xlim(0, top_c["vendas_por_mes_ativo"].max() * 1.48)

plt.tight_layout()
plt.savefig(FIG_DIR / "01_eda_bairros_e_canais.png", dpi=140)
plt.show()
"""),
        ("md", """### Principais Conclusões da EDA
1. **Calendário Ponderado funciona na prática**: a normalização por dias úteis ponderados explica imediatamente os movimentos de calendário de 2026 (ex.: Fev/2026 com `20,0 DU` caiu em volume bruto, mas **cresceu** em ritmo diário `78,5 vendas/DU`; Jul/2026 com `25,0 DU` teve pico bruto mantendo ritmo estável `83,4 vendas/DU`). Além disso, empiricamente o sábado da The Fiber é muito forte (~`0,74` de um dia útil).
2. **Anomalias de Retiradas**: `2025-06` (`1.710` retiradas) e `2025-08` (`1.065` retiradas) são baixas concentradas em inadimplência; precisam de tratamento explícito no treino da Regressão Linear.
3. **Alavanca de Novos Bairros**: bairros novos em **Rede Própria** maturam muito melhor (média de ~`9,0` vendas/mês no M6 vs. ~`3,3` em **Rede Neutra**), com menor volume relativo de retiradas.
4. **Alavanca de Novos Canais**: canais abertos entre o fim de 2024 e 2026 (`Loja Lorena`, `Loja Vera Cruz`, `Loja Maysa`, `INDIQUE E GANHE`) foram decisivos para elevar o patamar da empresa de ~`450` para ~`2.000` vendas/mês e com FPD substancialmente inferior ao canal legado `Agente Autorizado`."""),
    ]
    return _make_nb(cells)


def build_nb_02() -> nbf.NotebookNode:
    cells = [
        ("md", """# 02. Engenharia de Atributos e Calendário Operacional (2024–2027)

**Etapa CRISP-DM:** Preparação dos Dados (*Feature Engineering*)

### O que construímos nesta etapa
1. **Calendário Operacional 2024–2027** com feriados nacionais, estaduais de Goiás e municipais de Goiânia + Carnaval/Corpus Christi, aplicando a regra exata de pesos (`Dia Útil = 1,0`, `Sábado = 0,5`, `Domingo/Feriado = 0,0`).
2. **Variáveis estruturais e causais** (sem vazamento do futuro):
   - Para **Vendas**: `dias_uteis_ponderados`, `qtd_bairros_ativos_mes`, `qtd_canais_ativos_mes`, `vendas_por_du_lag1` / `vendas_lag1`, sazonalidade trigonométrica (`mes_sin`, `mes_cos`).
   - Para **Retiradas**: `dias_uteis_ponderados`, `carteira_inicio_mes` (base exposta ao churn), `vendas_acum_3m_ant` (safras jovens M1–M3 com maior propensão a churn/FPD), `retiradas_lag1` (limpo de anomalias de lote) e sazonalidade.
3. **Análise de Correlação e Multicolinearidade (VIF)** para orientar a escolha entre OLS puro e Regressão Linear Regularizada (Ridge / Lasso / ElasticNet)."""),
        ("code", """from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path("..").resolve()
DATA_PROC = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"

sns.set_theme(style="whitegrid")

df_cal_mensal = pd.read_parquet(DATA_PROC / "calendario_mensal.parquet")
df_ml = pd.read_parquet(DATA_PROC / "serie_mensal_ml.parquet")
df_closed = df_ml[df_ml["mes_fechado"]].copy()

print("=== CALENDÁRIO OPERACIONAL DE 2027 (PESOS 1,0 / 0,5 / 0,0) ===")
cal_2027 = df_cal_mensal[df_cal_mensal["ano"] == 2027].copy()
display(cal_2027[[
    "anomes", "dias_calendario", "dias_uteis_cheios", "sabados_uteis",
    "domingos", "feriados_em_dia_util", "dias_uteis_ponderados"
]])
print(f"Total de Dias Úteis Ponderados em 2027: {cal_2027['dias_uteis_ponderados'].sum():.1f}")
"""),
        ("md", """## 1. Matriz de Correlação — Drivers de Vendas e Retiradas

Verificamos as correlações lineares (Pearson) entre os alvos (`vendas`, `vendas_por_du`, `retiradas`) e as variáveis estruturais/causais nos meses fechados (excluindo os dois meses de baixa em lote ao avaliar retiradas)."""),
        ("code", """cols_corr = [
    "vendas", "vendas_por_du", "retiradas",
    "dias_uteis_ponderados", "qtd_bairros_ativos_mes", "qtd_canais_ativos_mes",
    "qtd_vendedores_ativos_mes", "carteira_inicio_mes", "vendas_acum_3m_ant",
    "vendas_lag1", "retiradas_lag1", "taxa_fpd_61d_lag3"
]
df_clean_corr = df_closed[df_closed["anomalia_retirada"] == 0][cols_corr].dropna()
corr_mat = df_clean_corr.corr()

fig, ax = plt.subplots(figsize=(11, 8))
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Matriz de Correlação de Pearson (Meses Fechados sem Anomalias de Lote)")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_matriz_correlacao.png", dpi=140)
plt.show()
"""),
        ("md", """## 2. Relações Lineares Principais e Diagnóstico de Multicolinearidade (VIF)

Um cuidado clássico em Regressão Linear é medir o **VIF (*Variance Inflation Factor*)**. Quando variáveis estruturais crescem juntas no tempo (ex.: `carteira_inicio_mes` e `vendas_acum_3m_ant`, ou `qtd_bairros_ativos_mes` e `vendas_lag1`), o VIF tende a ficar acima de `5–10`, tornando o **OLS puro instável**. É exatamente por isso que na etapa 03 comparamos OLS com **Ridge, Lasso e ElasticNet**."""),
        ("code", """def calc_vif_table(df_in: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df_in[cols].dropna().astype(float)
    X_const = np.column_stack([np.ones(len(X)), X.values])
    rows = []
    for i, c in enumerate(cols):
        rows.append({"feature": c, "VIF": round(float(variance_inflation_factor(X_const, i + 1)), 2)})
    return pd.DataFrame(rows)

print("=== VIF — Features de Vendas (Taxa por Dia Útil) ===")
display(calc_vif_table(df_closed, ["qtd_bairros_ativos_mes", "qtd_canais_ativos_mes", "vendas_por_du_lag1", "mes_sin", "mes_cos"]))

print("=== VIF — Features de Retiradas (Sem Anomalias de Lote) ===")
display(calc_vif_table(df_closed[df_closed["anomalia_retirada"] == 0], ["dias_uteis_ponderados", "carteira_inicio_mes", "vendas_acum_3m_ant", "retiradas_lag1", "mes_sin", "mes_cos"]))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.regplot(data=df_closed, x="qtd_bairros_ativos_mes", y="vendas_por_du", ax=axes[0], scatter_kws={"s": 60, "alpha": 0.8}, line_kws={"color": "#d62728"})
axes[0].set_title("Linearidade: Bairros Ativos no Mês vs. Vendas / Dia Útil Ponderado")
axes[0].set_xlabel("Quantidade de Bairros Ativos no Mês")
axes[0].set_ylabel("Vendas por Dia Útil Ponderado")

df_ret_ok = df_closed[df_closed["anomalia_retirada"] == 0]
sns.regplot(data=df_ret_ok, x="carteira_inicio_mes", y="retiradas", ax=axes[1], scatter_kws={"s": 60, "alpha": 0.8, "color": "#2ca02c"}, line_kws={"color": "#1b4f72"})
axes[1].set_title("Linearidade: Carteira Ativa (Início do Mês) vs. Retiradas Mensais")
axes[1].set_xlabel("Carteira Ativa no Início do Mês")
axes[1].set_ylabel("Retiradas no Mês (sem baixas em lote)")

plt.tight_layout()
plt.savefig(FIG_DIR / "02_relacoes_lineares_principais.png", dpi=140)
plt.show()
"""),
    ]
    return _make_nb(cells)


def build_nb_03() -> nbf.NotebookNode:
    cells = [
        ("md", """# 03. Modelagem de Regressão Linear, Validação Temporal e Diagnóstico

**Etapa CRISP-DM:** Modelagem e Avaliação (*MLOps Benchmark & Diagnostics*)

### Metodologia de Validação (sem Data Leakage)
- **Conjunto de Treino**: `2024-02` até `2026-06` (seleção de hiperparâmetros com *Expanding-Window* `TimeSeriesSplit`).
- **Conjunto de Validação Out-of-Time (Holdout 2026)**: últimos 4 meses fechados (`2026-07` a `2026-10`).
- **Modelos comparados (família Regressão Linear)**: `OLS`, `Ridge`, `Lasso`, `ElasticNet` e `HuberRegressor`, testando tanto a formulação **Taxa por Dia Útil Ponderado** (`y / DU * DU`) quanto **Volume Direto** (`dias_uteis_ponderados` como preditor), além do impacto explícito de tratar as anomalias de baixa em lote nas retiradas.
- Após escolher os campeões pelo erro fora da amostra, cada campeão é **re-treinado com 100% dos meses fechados** para gerar a projeção de 2027."""),
        ("code", """import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

ROOT = Path("..").resolve()
DATA_PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"

sns.set_theme(style="whitegrid")

with open(MODELS_DIR / "metadata.json", encoding="utf-8") as f:
    meta = json.load(f)

bench_v = pd.read_csv(MODELS_DIR / "benchmark_vendas.csv")
bench_r = pd.read_csv(MODELS_DIR / "benchmark_retiradas.csv")
df_ml = pd.read_parquet(DATA_PROC / "serie_mensal_ml.parquet")
df_closed = df_ml[df_ml["mes_fechado"]].copy()

print("=== BENCHMARK DE MODELOS LINEARES — VENDAS ===")
display(bench_v)

print("=== BENCHMARK DE MODELOS LINEARES — RETIRADAS ===")
display(bench_r)
"""),
        ("md", """## 1. Desempenho no Holdout Out-of-Time (Jul/2026 a Out/2026)

A tabela e os gráficos abaixo comparam o **Valor Real vs. Valor Previsto** pelos modelos campeões nos 4 meses de holdout que ficaram totalmente ocultos durante o treinamento."""),
        ("code", """val_v = pd.DataFrame(meta["champion_vendas"]["val_predictions"])
val_r = pd.DataFrame(meta["champion_retiradas"]["val_predictions"])

print(f"Campeão VENDAS: {meta['champion_vendas']['model_name']} | {meta['champion_vendas']['best_config']}")
print("Métricas Holdout 2026:", meta["champion_vendas"]["val_metrics"])
display(val_v)

print(f"\\nCampeão RETIRADAS: {meta['champion_retiradas']['model_name']} | {meta['champion_retiradas']['best_config']}")
print("Métricas Holdout 2026:", meta["champion_retiradas"]["val_metrics"])
display(val_r)

# Carregar artefatos finais para visualizar ajuste completo + holdout
art_v = joblib.load(MODELS_DIR / "modelo_vendas.joblib")
art_r = joblib.load(MODELS_DIR / "modelo_retiradas.joblib")

df_v_plot = df_closed.dropna(subset=art_v["features"]).copy()
pred_v_raw = art_v["pipeline"].predict(df_v_plot[art_v["features"]])
df_v_plot["vendas_fit"] = pred_v_raw * df_v_plot["dias_uteis_ponderados"] if art_v["scale_by_du"] else pred_v_raw

df_r_plot = df_closed.dropna(subset=art_r["features"]).copy()
pred_r_raw = art_r["pipeline"].predict(df_r_plot[art_r["features"]])
df_r_plot["retiradas_fit"] = pred_r_raw * df_r_plot["dias_uteis_ponderados"] if art_r["scale_by_du"] else pred_r_raw

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

# Vendas
axes[0].plot(df_v_plot["anomes"], df_v_plot["vendas"], marker="o", lw=2.2, label="Real — Vendas", color="#1f77b4")
axes[0].plot(df_v_plot["anomes"], df_v_plot["vendas_fit"], ls="--", lw=2.0, label=f"Ajuste {art_v['model_name']}", color="#ff7f0e")
axes[0].scatter(val_v["anomes"], val_v["previsto"], s=90, color="#2ca02c", zorder=5, label="Holdout Out-of-Time (2026-07 a 2026-10)")
axes[0].axvspan("2026-07", "2026-10", color="#2ca02c", alpha=0.10)
axes[0].set_title(f"Vendas — Aderência Histórica e Backtest 2026 (Holdout MAPE: {meta['champion_vendas']['val_metrics']['MAPE_pct']}%)")
axes[0].set_ylabel("Vendas / Mês")
axes[0].tick_params(axis="x", rotation=45)
axes[0].legend(loc="upper left")

# Retiradas
axes[1].plot(df_r_plot["anomes"], df_r_plot["retiradas"], marker="s", lw=2.0, label="Real — Retiradas", color="#d62728")
axes[1].plot(df_r_plot["anomes"], df_r_plot["retiradas_fit"], ls="--", lw=2.0, label=f"Ajuste {art_r['model_name']}", color="#1b4f72")
axes[1].scatter(val_r["anomes"], val_r["previsto"], s=90, color="#2ca02c", zorder=5, label="Holdout Out-of-Time (2026-07 a 2026-10)")
anom_r = df_r_plot[df_r_plot["anomalia_retirada"] == 1]
axes[1].scatter(anom_r["anomes"], anom_r["retiradas"], s=130, color="#ff7f0e", edgecolor="black", zorder=6, label="Anomalia excluída do treino")
axes[1].axvspan("2026-07", "2026-10", color="#2ca02c", alpha=0.10)
axes[1].set_title(f"Retiradas — Aderência Histórica e Backtest 2026 (Holdout MAPE: {meta['champion_retiradas']['val_metrics']['MAPE_pct']}%)")
axes[1].set_ylabel("Retiradas / Mês")
axes[1].tick_params(axis="x", rotation=45)
axes[1].legend(loc="upper left")

plt.tight_layout()
plt.savefig(FIG_DIR / "03_backtest_vendas_e_retiradas.png", dpi=140)
plt.show()
"""),
        ("md", """## 2. Interpretação dos Coeficientes e Diagnóstico Estatístico dos Resíduos

Uma das maiores vantagens da Regressão Linear é a **interpretabilidade executiva**: conseguimos inspecionar exatamente o peso de cada variável (tanto padronizada para importância relativa quanto na escala original para interpretação de negócio) e validar as premissas clássicas nos resíduos (normalidade via **Shapiro-Wilk** e autocorrelação via **Durbin-Watson**)."""),
        ("code", """coef_v = pd.DataFrame(meta["champion_vendas"]["production_fit"]["coefficients"]["features"])
coef_r = pd.DataFrame(meta["champion_retiradas"]["production_fit"]["coefficients"]["features"])

print("=== COEFICIENTES — MODELO CAMPEÃO DE VENDAS ===")
display(coef_v)
print("Diagnóstico de Resíduos (Vendas):", meta["champion_vendas"]["production_fit"]["diagnostics"])

print("\\n=== COEFICIENTES — MODELO CAMPEÃO DE RETIRADAS ===")
display(coef_r)
print("Diagnóstico de Resíduos (Retiradas):", meta["champion_retiradas"]["production_fit"]["diagnostics"])

# Gráficos de diagnóstico de resíduos
res_v = df_v_plot["vendas"] - df_v_plot["vendas_fit"]
df_r_clean = df_r_plot[df_r_plot["anomalia_retirada"] == 0]
res_r = df_r_clean["retiradas"] - df_r_clean["retiradas_fit"]

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

axes[0, 0].axhline(0, color="black", ls="--", lw=1)
axes[0, 0].scatter(df_v_plot["vendas_fit"], res_v, color="#1f77b4", alpha=0.8)
axes[0, 0].set_title("Vendas: Resíduos vs. Valores Ajustados")
axes[0, 0].set_xlabel("Vendas Ajustadas")
axes[0, 0].set_ylabel("Resíduo")

stats.probplot(res_v, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title("Vendas: Q-Q Plot dos Resíduos")

axes[1, 0].axhline(0, color="black", ls="--", lw=1)
axes[1, 0].scatter(df_r_clean["retiradas_fit"], res_r, color="#d62728", alpha=0.8)
axes[1, 0].set_title("Retiradas (sem lote): Resíduos vs. Valores Ajustados")
axes[1, 0].set_xlabel("Retiradas Ajustadas")
axes[1, 0].set_ylabel("Resíduo")

stats.probplot(res_r, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Retiradas (sem lote): Q-Q Plot dos Resíduos")

plt.tight_layout()
plt.savefig(FIG_DIR / "03_diagnostico_residuos.png", dpi=140)
plt.show()
"""),
    ]
    return _make_nb(cells)


def build_nb_04() -> nbf.NotebookNode:
    cells = [
        ("md", """# 04. Projeção 2027 e Simulador de Cenários de Crescimento

**Etapa CRISP-DM:** Entrega Executiva / Deployment (*Forecast Cascata 2027*)

### Como funciona o motor de projeção recursivo em cascata
Partindo do último mês fechado real (`2026-10`, carteira de **33.755** clientes):
1. Projeta os meses-ponte de encerramento de 2026 (`2026-11` e `2026-12`) igualmente em todos os cenários, garantindo partida idêntica em **01/01/2027** (`35.645` clientes).
2. Para cada mês de **Jan/2027 a Dez/2027**, combina o **calendário real de dias úteis ponderados (`1 / 0,5 / 0`)** com as alavancas de expansão (novos bairros ativos/mês e novos canais comerciais) em três cenários:
   - **Conservador**: manutenção da malha atual com expansão mínima de bairros (`+1/mês`), sem novos canais.
   - **Base**: continuidade orgânica de 2026 com expansão moderada de bairros (`+4/mês`).
   - **Expansão**: abertura acelerada de novos bairros (`+9/mês`) + entrada de **1 novo canal comercial** a partir de Mar/2027.
3. Acopla **Vendas -> Carteira Ativa & Safras Recentes (3m) -> Retiradas -> Saldo Líquido -> Nova Carteira** mês a mês, com **Intervalo de Predição de 90%**."""),
        ("code", """import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path("..").resolve()
DATA_PROC = ROOT / "data" / "processed"
DATA_PRED = ROOT / "data" / "predictions"
FIG_DIR = ROOT / "reports" / "figures"

sns.set_theme(style="whitegrid")

df_hist = pd.read_parquet(DATA_PROC / "serie_mensal_ml.parquet")
df_closed = df_hist[df_hist["mes_fechado"]].copy()
df_pred = pd.read_parquet(DATA_PRED / "projecao_2027.parquet")
with open(DATA_PRED / "resumo_2027.json", encoding="utf-8") as f:
    resumo = json.load(f)

print("=== QUADRO EXECUTIVO COMPARATIVO — CENÁRIOS 2027 ===")
df_resumo = pd.DataFrame(resumo["cenarios_2027"]).T
cols_show = [
    "vendas_total_2027", "vendas_media_mensal_2027",
    "retiradas_total_2027", "retiradas_media_mensal_2027",
    "saldo_liquido_2027", "carteira_inicio_2027", "carteira_fim_2027",
    "crescimento_carteira_2027_pct", "churn_medio_mensal_sobre_carteira_pct",
    "relacao_retirada_venda_2027_pct",
]
display(df_resumo[cols_show])
"""),
        ("md", """## 1. Projeção Mensal Detalhada de 2027 — Cenário Base

Observe como a ponderação de dias úteis (`1 / 0,5 / 0`) se reflete com naturalidade ao longo de 2027: meses mais curtos ou com feriados (como **Fev/2027** com `20,0 DU` no Carnaval e **Mai/2027** com `21,0 DU`) projetam volumes proporcionais à capacidade útil real, enquanto meses longos (**Mar, Jun, Ago, Dez**) concentram maior volume."""),
        ("code", """base_2027 = df_pred[(df_pred["cenario"] == "Base") & (df_pred["ano"] == 2027)].copy()
display(base_2027[[
    "anomes", "dias_uteis_ponderados", "qtd_bairros_ativos_proj",
    "vendas_prev", "vendas_ic_inf", "vendas_ic_sup",
    "retiradas_prev", "saldo_liquido_prev", "carteira_fim_mes_prev",
    "taxa_churn_carteira_pct", "relacao_retirada_venda_pct"
]].reset_index(drop=True))
"""),
        ("md", """## 2. Gráficos Executivos — Trajetória 2024–2027 (Vendas, Retiradas e Crescimento de Carteira)"""),
        ("code", """base_full = df_pred[df_pred["cenario"] == "Base"].copy()
cons_full = df_pred[df_pred["cenario"] == "Conservador"].copy()
exp_full = df_pred[df_pred["cenario"] == "Expansao"].copy()

fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Painel 1: Histórico + Projeção Base de Vendas e Retiradas com IC 90%
ax = axes[0]
ax.plot(df_closed["anomes"], df_closed["vendas"], color="#1f77b4", lw=2.2, marker="o", markersize=4, label="Vendas Reais (2024–2026)")
ax.plot(base_full["anomes"], base_full["vendas_prev"], color="#1f77b4", lw=2.4, ls="--", marker="o", markersize=4, label="Vendas Projetadas (Base)")
ax.fill_between(base_full["anomes"], base_full["vendas_ic_inf"], base_full["vendas_ic_sup"], color="#1f77b4", alpha=0.15, label="IC 90% Vendas")

ax.plot(df_closed["anomes"], df_closed["retiradas"], color="#d62728", lw=1.8, marker="s", markersize=4, label="Retiradas Reais (2024–2026)")
ax.plot(base_full["anomes"], base_full["retiradas_prev"], color="#d62728", lw=2.4, ls="--", marker="s", markersize=4, label="Retiradas Projetadas (Base)")
ax.fill_between(base_full["anomes"], base_full["retiradas_ic_inf"], base_full["retiradas_ic_sup"], color="#d62728", alpha=0.14, label="IC 90% Retiradas")

ax.axvline("2027-01", color="black", ls=":", lw=1.4)
ax.text("2027-01", ax.get_ylim()[1] * 0.92, "  Início 2027", fontweight="bold", fontsize=10)
ax.set_title("Histórico (2024–2026) e Projeção 2027 — Vendas vs. Retiradas (Cenário Base, IC 90%)")
ax.set_ylabel("Contratos / Mês")
ax.tick_params(axis="x", rotation=55)
ticks_all = list(df_closed["anomes"]) + list(base_full["anomes"])
ax.set_xticks(ticks_all[::2])
ax.legend(loc="upper left", ncol=2, fontsize=9)

# Painel 2: Curva de Crescimento da Carteira (Planta) nos 3 cenários
ax2 = axes[1]
ax2.plot(df_closed["anomes"], df_closed["carteira_fim_mes"], color="#1b4f72", lw=2.6, label="Carteira Real (Planta 2024–2026)")
ax2.plot(exp_full["anomes"], exp_full["carteira_fim_mes_prev"], color="#2ca02c", lw=2.4, ls="--", marker="^", markersize=4,
         label=f"Expansão 2027 ({int(exp_full.iloc[-1]['carteira_fim_mes_prev']):,} | +{resumo['cenarios_2027']['Expansao']['crescimento_carteira_2027_pct']:.1f}%)")
ax2.plot(base_full["anomes"], base_full["carteira_fim_mes_prev"], color="#1f77b4", lw=2.5, ls="--", marker="o", markersize=4,
         label=f"Base 2027 ({int(base_full.iloc[-1]['carteira_fim_mes_prev']):,} | +{resumo['cenarios_2027']['Base']['crescimento_carteira_2027_pct']:.1f}%)")
ax2.plot(cons_full["anomes"], cons_full["carteira_fim_mes_prev"], color="#ff7f0e", lw=2.4, ls="--", marker="v", markersize=4,
         label=f"Conservador 2027 ({int(cons_full.iloc[-1]['carteira_fim_mes_prev']):,} | +{resumo['cenarios_2027']['Conservador']['crescimento_carteira_2027_pct']:.1f}%)")

ax2.axvline("2027-01", color="black", ls=":", lw=1.4)
ax2.set_title("Crescimento Projetado da Carteira Ativa (Planta) até Dezembro/2027 — Comparativo de Cenários")
ax2.set_ylabel("Clientes Ativos na Planta")
ax2.tick_params(axis="x", rotation=55)
ax2.set_xticks(ticks_all[::2])
ax2.legend(loc="upper left", fontsize=10)

plt.tight_layout()
plt.savefig(FIG_DIR / "04_projecao_2027_cenarios_carteira.png", dpi=140)
plt.show()
"""),
        ("md", """## 3. Saldo Líquido Mensal de 2027 e Comparativo Histórico Anual (2024–2027)"""),
        ("code", """fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))

# Saldo líquido mensal 2027 nos 3 cenários
df_2027_all = df_pred[df_pred["ano"] == 2027].copy()
sns.barplot(data=df_2027_all, x="anomes", y="saldo_liquido_prev", hue="cenario",
            palette={"Conservador": "#ff7f0e", "Base": "#1f77b4", "Expansao": "#2ca02c"}, ax=axes[0])
axes[0].set_title("Saldo Líquido Mensal Projetado em 2027 (Vendas - Retiradas)")
axes[0].set_xlabel("")
axes[0].set_ylabel("Crescimento Líquido (Contratos / Mês)")
axes[0].tick_params(axis="x", rotation=45)
axes[0].legend(title="Cenário")

# Evolução Anual Consolidada: 2024 Real, 2025 Real, 2026 Real+Ponte, 2027 Base
hist_anual = df_closed.groupby("ano", as_index=False).agg(vendas=("vendas", "sum"), retiradas=("retiradas", "sum"))
ponte_2026 = base_full[base_full["ano"] == 2026]
v_2026_full = float(hist_anual.loc[hist_anual["ano"] == 2026, "vendas"].iloc[0] + ponte_2026["vendas_prev"].sum())
r_2026_full = float(hist_anual.loc[hist_anual["ano"] == 2026, "retiradas"].iloc[0] + ponte_2026["retiradas_prev"].sum())

df_comp = pd.DataFrame([
    {"Periodo": "2024 (Real)", "Vendas": float(hist_anual.loc[hist_anual["ano"] == 2024, "vendas"].iloc[0]), "Retiradas": float(hist_anual.loc[hist_anual["ano"] == 2024, "retiradas"].iloc[0])},
    {"Periodo": "2025 (Real)", "Vendas": float(hist_anual.loc[hist_anual["ano"] == 2025, "vendas"].iloc[0]), "Retiradas": float(hist_anual.loc[hist_anual["ano"] == 2025, "retiradas"].iloc[0])},
    {"Periodo": "2026 (Real+Ponte)", "Vendas": round(v_2026_full, 0), "Retiradas": round(r_2026_full, 0)},
    {"Periodo": "2027 (Conserv.)", "Vendas": resumo["cenarios_2027"]["Conservador"]["vendas_total_2027"], "Retiradas": resumo["cenarios_2027"]["Conservador"]["retiradas_total_2027"]},
    {"Periodo": "2027 (Base)", "Vendas": resumo["cenarios_2027"]["Base"]["vendas_total_2027"], "Retiradas": resumo["cenarios_2027"]["Base"]["retiradas_total_2027"]},
    {"Periodo": "2027 (Expansão)", "Vendas": resumo["cenarios_2027"]["Expansao"]["vendas_total_2027"], "Retiradas": resumo["cenarios_2027"]["Expansao"]["retiradas_total_2027"]},
])
df_comp["Saldo_Liquido"] = df_comp["Vendas"] - df_comp["Retiradas"]
display(df_comp)

df_comp_m = df_comp.melt(id_vars="Periodo", value_vars=["Vendas", "Retiradas", "Saldo_Liquido"], var_name="Indicador", value_name="Contratos")
sns.barplot(data=df_comp_m, x="Periodo", y="Contratos", hue="Indicador",
            palette={"Vendas": "#1f77b4", "Retiradas": "#d62728", "Saldo_Liquido": "#2ca02c"}, ax=axes[1])
axes[1].set_title("Comparativo Anual Consolidado (2024–2027)")
axes[1].set_xlabel("")
axes[1].set_ylabel("Contratos no Ano")
axes[1].tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.savefig(FIG_DIR / "04_saldo_liquido_e_comparativo_anual.png", dpi=140)
plt.show()
"""),
        ("md", """### Conclusões Executivas & Próximos Passos de MLOps
1. **Crescimento consistente em 2027**: no cenário **Base**, a empresa projeta **24.430 vendas** contra **13.917 retiradas**, gerando **+10.513 clientes líquidos** e elevando a planta de **35.645** para **46.157 clientes (+29,5%)**.
2. **Impacto real de Novos Bairros e Novos Canais**: no cenário **Expansão** (priorizando abertura de bairros de **Rede Própria** — que a EDA mostrou entregar ~9 vendas/mês no M6 — somado a 1 novo canal comercial qualificado), a carteira pode alcançar **51.742 clientes (+45,2%)** com relação retirada/venda caindo para **48,3%**.
3. **Rotina MLOps recomendada**:
   - Re-executar `python -m src.pipeline --step all` todo início de mês após o fechamento do mês anterior no Airflow;
   - Monitorar o erro mensal (`MAPE`) real vs. previsto em `models/metadata.json` para detectar *drift* comercial ou de inadimplência;
   - Opcionalmente, expor `data/predictions/projecao_2027.json` em uma nova aba **"Projeção 2027"** dentro do `dash-retencao`."""),
    ]
    return _make_nb(cells)


def main() -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    builders = {
        "01_analise_exploratoria_eda.ipynb": build_nb_01,
        "02_engenharia_de_atributos.ipynb": build_nb_02,
        "03_modelagem_regressao_linear.ipynb": build_nb_03,
        "04_projecao_2027_e_cenarios.ipynb": build_nb_04,
    }
    for fname, fn in builders.items():
        nb = fn()
        out_path = NOTEBOOKS_DIR / fname
        with open(out_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Gerado: {out_path}")


if __name__ == "__main__":
    main()
