"""
Engenharia de atributos e construção das bases analíticas de ML.

Gera:
  - data/processed/calendario_diario.parquet & calendario_mensal.parquet
  - data/processed/serie_diaria.parquet        (validação empírica dos pesos 1 / 0,5 / 0)
  - data/processed/serie_mensal_ml.parquet     (base principal de treino e backtest)
  - data/processed/analise_bairros.parquet     (EDA de novos bairros e curva de ramp-up)
  - data/processed/analise_canais.parquet      (EDA de canais e entrada de novos canais)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, ensure_directories, load_params
from src.features.calendar import build_calendar, evaluate_empirical_day_weights


def _detect_closed_months(df_contracts: pd.DataFrame) -> str:
    """
    Identifica o último mês completamente fechado nos dados.
    Se a data máxima de ativação/cancelamento estiver antes do último dia do mês,
    considera o mês anterior como o último mês fechado.
    """
    datas_atv = pd.to_datetime(df_contracts["data_ativacao"], errors="coerce").dropna()
    datas_canc = pd.to_datetime(df_contracts["data_cancelamento"], errors="coerce").dropna()
    max_dt = min(datas_atv.max(), datas_canc.max())

    # Verifica se max_dt está nos últimos 2 dias do mês (tolerância D-1)
    month_end = (max_dt + pd.offsets.MonthEnd(0)).normalize()
    if (month_end - max_dt.normalize()).days <= 2:
        return max_dt.strftime("%Y-%m")
    prev_month = (max_dt.replace(day=1) - pd.Timedelta(days=1))
    return prev_month.strftime("%Y-%m")


def _mad_zscore(series: pd.Series) -> pd.Series:
    """Calcula Z-score robusto baseado na Mediana e MAD (imune a extremos)."""
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0 or pd.isna(mad):
        return pd.Series(0.0, index=series.index)
    return 0.6745 * (series - median) / mad


def build_features(
    raw_dir: Path = DATA_RAW_DIR,
    processed_dir: Path = DATA_PROCESSED_DIR,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Constrói todas as tabelas analíticas e de modelagem."""
    ensure_directories()
    params = params or load_params()
    data_cfg = params["data"]
    start_month = data_cfg["start_month"]
    base_cutoff = data_cfg["planta_base_cutoff"]
    excl_padrao = bool(data_cfg.get("exclude_vendedor_padrao_vendas", True))

    df = pd.read_parquet(raw_dir / "contratos_tratados.parquet")

    # 1. Calendário 2024–2027
    df_cal_diario, df_cal_mensal = build_calendar(params)
    df_cal_diario.to_parquet(processed_dir / "calendario_diario.parquet", index=False)
    df_cal_mensal.to_parquet(processed_dir / "calendario_mensal.parquet", index=False)

    last_closed_month = _detect_closed_months(df)

    # 2. Filtros padrão alinhados ao dash-retencao
    df_valid_vendedor = df[~df["ignorar_vendedor"]].copy()

    # Ativações (Vendas)
    mask_atv = df_valid_vendedor["flag_ativacao"] == 1
    if excl_padrao:
        mask_atv = mask_atv & (~df_valid_vendedor["is_vendedor_padrao"])
    df_atv_all = df_valid_vendedor[mask_atv & df_valid_vendedor["anomes_ativacao"].notna()].copy()

    # Retiradas (Todos os cancelamentos válidos no mês, independente da safra)
    mask_canc = (
        (df_valid_vendedor["flag_churn"] == 1)
        & (~df_valid_vendedor["ignorar_motivo"])
        & df_valid_vendedor["anomes_cancelamento"].notna()
    )
    df_canc_all = df_valid_vendedor[mask_canc].copy()

    # 3. Base inicial da carteira (Planta ativa até 2023-12, espelhando get_planta_evolucao_data)
    df_planta_base = df_valid_vendedor[df_valid_vendedor["anomes_ativacao"].notna()].copy()
    is_churn_planta = (df_planta_base["flag_churn"] == 1) & (~df_planta_base["ignorar_motivo"])
    cond_base_inicial = (df_planta_base["anomes_ativacao"] <= base_cutoff) & (
        (~is_churn_planta)
        | (
            df_planta_base["anomes_cancelamento"].notna()
            & (df_planta_base["anomes_cancelamento"] > base_cutoff)
        )
    )
    carteira_inicial_base = int(cond_base_inicial.sum())

    # 4. Série diária (para validação empírica dos pesos 1 / 0,5 / 0)
    atv_d = df_atv_all[df_atv_all["anomes_ativacao"] >= start_month].copy()
    atv_d["data"] = pd.to_datetime(atv_d["data_ativacao"], errors="coerce").dt.normalize()
    atv_daily = atv_d.dropna(subset=["data"]).groupby("data").size().rename("vendas")

    canc_d = df_canc_all[df_canc_all["anomes_cancelamento"] >= start_month].copy()
    canc_d["data"] = pd.to_datetime(canc_d["data_cancelamento"], errors="coerce").dt.normalize()
    canc_daily = canc_d.dropna(subset=["data"]).groupby("data").size().rename("retiradas")

    max_closed_dt = pd.Timestamp(f"{last_closed_month}-01") + pd.offsets.MonthEnd(0)
    df_serie_diaria = df_cal_diario[
        (df_cal_diario["data"] >= pd.Timestamp(f"{start_month}-01"))
        & (df_cal_diario["data"] <= max_closed_dt)
    ].copy()
    df_serie_diaria["vendas"] = df_serie_diaria["data"].map(atv_daily).fillna(0).astype(int)
    df_serie_diaria["retiradas"] = df_serie_diaria["data"].map(canc_daily).fillna(0).astype(int)
    df_serie_diaria.to_parquet(processed_dir / "serie_diaria.parquet", index=False)

    emp_weights = evaluate_empirical_day_weights(df_serie_diaria)
    emp_weights.to_parquet(processed_dir / "pesos_empiricos_dias.parquet", index=False)

    # 5. Primeiro mês histórico de cada bairro e de cada canal (usando histórico completo!)
    bairro_first_month = (
        df_atv_all[df_atv_all["bairro_norm"] != "NÃO INFORMADO"]
        .groupby(["cidade_norm", "bairro_norm"])["anomes_ativacao"]
        .min()
        .reset_index()
        .rename(columns={"anomes_ativacao": "mes_estreia_bairro"})
    )
    canal_first_month = (
        df_atv_all[~df_atv_all["canal_venda_efetivo"].isin(["Não vinculado", "Não identificado"])]
        .groupby("canal_venda_efetivo")["anomes_ativacao"]
        .min()
        .reset_index()
        .rename(columns={"anomes_ativacao": "mes_estreia_canal"})
    )

    df_atv_period = df_atv_all[df_atv_all["anomes_ativacao"] >= start_month].copy()
    df_atv_period = df_atv_period.merge(
        bairro_first_month, on=["cidade_norm", "bairro_norm"], how="left"
    )
    df_atv_period["is_novo_bairro"] = (
        df_atv_period["anomes_ativacao"] == df_atv_period["mes_estreia_bairro"]
    )
    df_atv_period = df_atv_period.merge(
        canal_first_month, on="canal_venda_efetivo", how="left"
    )
    df_atv_period["is_novo_canal"] = (
        df_atv_period["anomes_ativacao"] == df_atv_period["mes_estreia_canal"]
    )

    # 6. Agregação mensal de Vendas + Expansão (Bairros/Canais/Rede) + FPD
    vendas_mensal = (
        df_atv_period.groupby("anomes_ativacao", as_index=False)
        .agg(
            vendas=("id_contrato", "count"),
            qtd_bairros_ativos_mes=("bairro_norm", "nunique"),
            qtd_canais_ativos_mes=("canal_venda_efetivo", "nunique"),
            qtd_vendedores_ativos_mes=("nome_vendedor", "nunique"),
            vendas_rede_propria=("tipo_rede_efetivo", lambda s: int((s == "Rede Própria").sum())),
            vendas_rede_neutra=("tipo_rede_efetivo", lambda s: int((s == "Rede Neutra").sum())),
            vendas_novos_bairros_mes=("is_novo_bairro", "sum"),
            fpd_61d_qtd=("flag_fpd_61d", lambda s: int((s == 1).sum())),
            fpd_com_os_qtd=(
                "qtd_os_fpd",
                lambda s: int((s.fillna(0) > 0).sum()),
            ),
        )
        .rename(columns={"anomes_ativacao": "anomes"})
    )

    novos_bairros_por_mes = (
        bairro_first_month[bairro_first_month["mes_estreia_bairro"] >= start_month]
        .groupby("mes_estreia_bairro")
        .size()
        .rename("novos_bairros_mes")
    )
    novos_canais_por_mes = (
        canal_first_month[canal_first_month["mes_estreia_canal"] >= start_month]
        .groupby("mes_estreia_canal")
        .size()
        .rename("novos_canais_mes")
    )

    # 7. Agregação mensal de Retiradas (Cancelamentos totais no mês, independente da safra)
    df_canc_period = df_canc_all[df_canc_all["anomes_cancelamento"] >= start_month].copy()
    retiradas_mensal = (
        df_canc_period.groupby("anomes_cancelamento", as_index=False)
        .agg(
            retiradas=("id_contrato", "count"),
            retiradas_inadimplencia=(
                "agrupamento_motivo_efetivo",
                lambda s: int((s == "Inadimplência").sum()),
            ),
            retiradas_voluntario=(
                "agrupamento_motivo_efetivo",
                lambda s: int((s == "Cancelamento Voluntário").sum()),
            ),
            retiradas_outros=(
                "agrupamento_motivo_efetivo",
                lambda s: int((~s.isin(["Inadimplência", "Cancelamento Voluntário"])).sum()),
            ),
        )
        .rename(columns={"anomes_cancelamento": "anomes"})
    )

    # 8. Montagem da série mensal completa (2024-01 até o último mês presente no DW)
    max_mes_dw = max(vendas_mensal["anomes"].max(), retiradas_mensal["anomes"].max())
    df_ml = df_cal_mensal[
        (df_cal_mensal["anomes"] >= start_month) & (df_cal_mensal["anomes"] <= max_mes_dw)
    ].copy()

    df_ml = df_ml.merge(vendas_mensal, on="anomes", how="left")
    df_ml = df_ml.merge(retiradas_mensal, on="anomes", how="left")
    df_ml["novos_bairros_mes"] = df_ml["anomes"].map(novos_bairros_por_mes).fillna(0).astype(int)
    df_ml["novos_canais_mes"] = df_ml["anomes"].map(novos_canais_por_mes).fillna(0).astype(int)

    int_cols = [
        "vendas",
        "retiradas",
        "retiradas_inadimplencia",
        "retiradas_voluntario",
        "retiradas_outros",
        "qtd_bairros_ativos_mes",
        "qtd_canais_ativos_mes",
        "qtd_vendedores_ativos_mes",
        "vendas_rede_propria",
        "vendas_rede_neutra",
        "vendas_novos_bairros_mes",
        "fpd_61d_qtd",
        "fpd_com_os_qtd",
    ]
    for col in int_cols:
        df_ml[col] = df_ml[col].fillna(0).astype(int)

    # Flag de mês fechado vs mês em andamento
    df_ml["mes_fechado"] = df_ml["anomes"] <= last_closed_month

    # Evolução da carteira (Planta) e crescimento líquido
    df_ml["saldo_liquido"] = df_ml["vendas"] - df_ml["retiradas"]
    carteira_ini = []
    carteira_fim = []
    curr_base = carteira_inicial_base
    for saldo in df_ml["saldo_liquido"]:
        carteira_ini.append(curr_base)
        curr_base = curr_base + int(saldo)
        carteira_fim.append(curr_base)
    df_ml["carteira_inicio_mes"] = carteira_ini
    df_ml["carteira_fim_mes"] = carteira_fim

    # Bairros acumulados na planta
    bairros_antes = int((bairro_first_month["mes_estreia_bairro"] < start_month).sum())
    df_ml["qtd_bairros_acumulados"] = bairros_antes + df_ml["novos_bairros_mes"].cumsum()

    # Taxas normalizadas por dia útil ponderado (regra 1 / 0,5 / 0)
    df_ml["vendas_por_du"] = df_ml["vendas"] / df_ml["dias_uteis_ponderados"]
    df_ml["retiradas_por_du"] = df_ml["retiradas"] / df_ml["dias_uteis_ponderados"]
    df_ml["taxa_churn_sobre_carteira"] = df_ml["retiradas"] / df_ml["carteira_inicio_mes"]
    df_ml["taxa_churn_sobre_vendas"] = np.where(
        df_ml["vendas"] > 0, df_ml["retiradas"] / df_ml["vendas"], np.nan
    )
    df_ml["share_rede_propria"] = np.where(
        df_ml["vendas"] > 0, df_ml["vendas_rede_propria"] / df_ml["vendas"], 0.0
    )
    df_ml["share_rede_neutra"] = np.where(
        df_ml["vendas"] > 0, df_ml["vendas_rede_neutra"] / df_ml["vendas"], 0.0
    )

    # FPD maturado (exclui censura à direita dos meses recentes < 61 dias)
    fpd_lag_m = int(data_cfg.get("fpd_maturation_lag_months", 3))
    closed_months_list = df_ml.loc[df_ml["mes_fechado"], "anomes"].tolist()
    fpd_cutoff_month = (
        closed_months_list[-fpd_lag_m] if len(closed_months_list) >= fpd_lag_m else closed_months_list[0]
    )
    df_ml["fpd_maturado"] = df_ml["anomes"] < fpd_cutoff_month
    df_ml["taxa_fpd_61d"] = np.where(
        df_ml["vendas"] > 0, df_ml["fpd_61d_qtd"] / df_ml["vendas"], np.nan
    )
    df_ml["taxa_fpd_61d_maturado"] = np.where(df_ml["fpd_maturado"], df_ml["taxa_fpd_61d"], np.nan)

    # Detecção transparente de anomalias de retirada (baixas em lote no IXC)
    # Combina razão contra mediana local centrada (5 meses) + Z-score MAD sobre taxa de churn
    closed_mask = df_ml["mes_fechado"]
    z_thresh = float(params["anomalies"]["churn_mad_zscore_threshold"])
    ratio_thresh = float(params["anomalies"].get("churn_local_ratio_threshold", 1.65))
    df_ml["churn_mad_zscore"] = 0.0
    df_ml["churn_local_ratio"] = 1.0
    closed_ret = df_ml.loc[closed_mask, "retiradas"].astype(float)
    local_med = closed_ret.rolling(window=5, center=True, min_periods=3).median()
    df_ml.loc[closed_mask, "churn_local_ratio"] = np.where(local_med > 0, closed_ret / local_med, 1.0)
    df_ml.loc[closed_mask, "churn_mad_zscore"] = _mad_zscore(
        df_ml.loc[closed_mask, "taxa_churn_sobre_carteira"]
    )
    df_ml["anomalia_retirada"] = (
        closed_mask
        & (
            (df_ml["churn_mad_zscore"] > z_thresh)
            | (df_ml["churn_local_ratio"] >= ratio_thresh)
        )
    ).astype(int)

    # Versão limpa (sem picos de baixa em lote) para lags de retiradas não contaminarem o mês seguinte
    retiradas_limpa = df_ml["retiradas"].astype(float).copy()
    retiradas_limpa[df_ml["anomalia_retirada"] == 1] = np.nan
    df_ml["retiradas_sem_anomalia_interp"] = retiradas_limpa.interpolate(method="linear")

    # Features temporais e sazonais
    df_ml["t"] = np.arange(len(df_ml), dtype=float)
    df_ml["mes_sin"] = np.sin(2 * np.pi * df_ml["mes"] / 12.0)
    df_ml["mes_cos"] = np.cos(2 * np.pi * df_ml["mes"] / 12.0)

    # Lags estritamente causais (apenas passado)
    df_ml["vendas_lag1"] = df_ml["vendas"].shift(1)
    df_ml["vendas_lag2"] = df_ml["vendas"].shift(2)
    df_ml["vendas_lag3"] = df_ml["vendas"].shift(3)
    df_ml["vendas_por_du_lag1"] = df_ml["vendas_por_du"].shift(1)
    df_ml["vendas_por_du_ma3"] = df_ml["vendas_por_du"].shift(1).rolling(3, min_periods=1).mean()
    df_ml["vendas_ma3"] = df_ml["vendas"].shift(1).rolling(3, min_periods=1).mean()
    df_ml["vendas_acum_3m_ant"] = (
        df_ml["vendas"].shift(1) + df_ml["vendas"].shift(2) + df_ml["vendas"].shift(3)
    )
    df_ml["retiradas_lag1"] = df_ml["retiradas_sem_anomalia_interp"].shift(1)
    df_ml["retiradas_por_du_lag1"] = (
        df_ml["retiradas_sem_anomalia_interp"] / df_ml["dias_uteis_ponderados"]
    ).shift(1)
    df_ml["retiradas_ma3"] = (
        df_ml["retiradas_sem_anomalia_interp"].shift(1).rolling(3, min_periods=1).mean()
    )
    df_ml["fpd_61d_qtd_lag3"] = df_ml["fpd_61d_qtd"].shift(3)
    df_ml["taxa_fpd_61d_lag3"] = df_ml["taxa_fpd_61d"].shift(3)
    df_ml["bairros_ativos_lag1"] = df_ml["qtd_bairros_ativos_mes"].shift(1)

    df_ml.to_parquet(processed_dir / "serie_mensal_ml.parquet", index=False)

    # 9. Tabela analítica de Bairros (para EDA de novos bairros e curva de ramp-up)
    b_atv = (
        df_atv_period[df_atv_period["anomes_ativacao"] <= last_closed_month]
        .groupby(["cidade_norm", "bairro_norm", "tipo_rede_efetivo", "mes_estreia_bairro", "anomes_ativacao"], as_index=False)
        .agg(vendas=("id_contrato", "count"), fpd_61d=("flag_fpd_61d", lambda s: int((s == 1).sum())))
        .rename(columns={"anomes_ativacao": "anomes"})
    )
    b_canc = (
        df_canc_period[df_canc_period["anomes_cancelamento"] <= last_closed_month]
        .groupby(["cidade_norm", "bairro_norm", "anomes_cancelamento"], as_index=False)
        .agg(retiradas=("id_contrato", "count"))
        .rename(columns={"anomes_cancelamento": "anomes"})
    )
    df_bairros_eda = b_atv.merge(b_canc, on=["cidade_norm", "bairro_norm", "anomes"], how="left")
    df_bairros_eda["retiradas"] = df_bairros_eda["retiradas"].fillna(0).astype(int)
    # Meses desde a estreia do bairro (idade do bairro na base: M0, M1, M2...)
    y_m = df_bairros_eda["anomes"].str.slice(0, 4).astype(int) * 12 + df_bairros_eda["anomes"].str.slice(5, 7).astype(int)
    y_e = df_bairros_eda["mes_estreia_bairro"].str.slice(0, 4).astype(int) * 12 + df_bairros_eda["mes_estreia_bairro"].str.slice(5, 7).astype(int)
    df_bairros_eda["idade_bairro_meses"] = (y_m - y_e).clip(lower=0)
    df_bairros_eda["bairro_novo_periodo"] = df_bairros_eda["mes_estreia_bairro"] >= start_month
    df_bairros_eda.to_parquet(processed_dir / "analise_bairros.parquet", index=False)

    # 10. Tabela analítica de Canais de Venda (para EDA de novos canais)
    c_atv = (
        df_atv_period[df_atv_period["anomes_ativacao"] <= last_closed_month]
        .groupby(["canal_venda_efetivo", "mes_estreia_canal", "anomes_ativacao"], as_index=False)
        .agg(
            vendas=("id_contrato", "count"),
            vendedores_ativos=("nome_vendedor", "nunique"),
            fpd_61d=("flag_fpd_61d", lambda s: int((s == 1).sum())),
        )
        .rename(columns={"anomes_ativacao": "anomes"})
    )
    c_canc = (
        df_canc_period[df_canc_period["anomes_cancelamento"] <= last_closed_month]
        .groupby(["canal_venda_efetivo", "anomes_cancelamento"], as_index=False)
        .agg(
            retiradas=("id_contrato", "count"),
            retiradas_inad=("agrupamento_motivo_efetivo", lambda s: int((s == "Inadimplência").sum())),
            retiradas_vol=("agrupamento_motivo_efetivo", lambda s: int((s == "Cancelamento Voluntário").sum())),
        )
        .rename(columns={"anomes_cancelamento": "anomes"})
    )
    df_canais_eda = c_atv.merge(c_canc, on=["canal_venda_efetivo", "anomes"], how="left")
    for col in ("retiradas", "retiradas_inad", "retiradas_vol"):
        df_canais_eda[col] = df_canais_eda[col].fillna(0).astype(int)
    df_canais_eda.to_parquet(processed_dir / "analise_canais.parquet", index=False)

    anomalias_detectadas = df_ml.loc[df_ml["anomalia_retirada"] == 1, ["anomes", "retiradas", "churn_mad_zscore"]].to_dict(
        orient="records"
    )

    summary = {
        "start_month": start_month,
        "last_closed_month": last_closed_month,
        "max_month_in_dw": max_mes_dw,
        "closed_months_count": int(df_ml["mes_fechado"].sum()),
        "carteira_inicial_2023_12": carteira_inicial_base,
        "carteira_ultimo_mes_fechado": int(df_ml.loc[df_ml["anomes"] == last_closed_month, "carteira_fim_mes"].iloc[0]),
        "fpd_cutoff_month": fpd_cutoff_month,
        "anomalias_retirada_detectadas": anomalias_detectadas,
        "pesos_empiricos": emp_weights.to_dict(orient="records"),
    }
    with open(processed_dir / "features_metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


if __name__ == "__main__":
    res = build_features()
    print(json.dumps(res, ensure_ascii=False, indent=2))
