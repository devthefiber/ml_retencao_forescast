"""
Motor de projeção recursiva em cascata para 2027 (Vendas, Retiradas e Crescimento de Carteira).

Parte do último mês fechado (ex.: 2026-10), projeta os meses-ponte restantes de 2026
e gera as projeções mês a mês de Janeiro a Dezembro de 2027 sob múltiplos cenários
(Conservador, Base e Expansão), respeitando:
  - Calendário real de 2027 com pesos (1,0 útil / 0,5 sábado / 0,0 domingo e feriado)
  - Acoplamento causal: Vendas -> Carteira Ativa & Safra recente (3m) -> Retiradas -> Saldo Líquido
  - Intervalo de predição calibrado com erro fora da amostra (90% IC)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import (
    DATA_PREDICTIONS_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    ensure_directories,
    load_params,
)


def _predict_single_step(artifact: dict[str, Any], row_dict: dict[str, float]) -> float:
    pipe = artifact["pipeline"]
    feats = artifact["features"]
    X = pd.DataFrame([{f: float(row_dict[f]) for f in feats}])
    raw_pred = float(pipe.predict(X)[0])
    if artifact["scale_by_du"]:
        vol = raw_pred * float(row_dict["dias_uteis_ponderados"])
    else:
        vol = raw_pred
    return max(0.0, vol)


def forecast_2027(
    processed_dir: Path = DATA_PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
    predictions_dir: Path = DATA_PREDICTIONS_DIR,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gera a projeção mensal para os meses-ponte de 2026 e todo o ano de 2027."""
    ensure_directories()
    params = params or load_params()

    df_hist = pd.read_parquet(processed_dir / "serie_mensal_ml.parquet")
    df_cal = pd.read_parquet(processed_dir / "calendario_mensal.parquet")
    df_closed = df_hist[df_hist["mes_fechado"]].sort_values("anomes").reset_index(drop=True)

    art_vendas = joblib.load(models_dir / "modelo_vendas.joblib")
    art_retiradas = joblib.load(models_dir / "modelo_retiradas.joblib")

    last_closed = df_closed.iloc[-1]
    last_closed_month = str(last_closed["anomes"])
    target_year = int(params["forecast"]["target_year"])
    conf_level = float(params["forecast"].get("confidence_level", 0.90))
    z_ci = float(norm.ppf(0.5 + conf_level / 2.0))

    # Horizonte futuro: do mês seguinte ao último mês fechado até Dezembro do target_year
    df_future_cal = df_cal[
        (df_cal["anomes"] > last_closed_month) & (df_cal["ano"] <= target_year)
    ].sort_values("anomes").reset_index(drop=True)

    # Patamares de partida baseados na média dos últimos 3 meses fechados
    recent_3 = df_closed.tail(3)
    base_bairros_ativos = float(recent_3["qtd_bairros_ativos_mes"].mean())
    base_canais_ativos = float(recent_3["qtd_canais_ativos_mes"].mean())

    sigma_v = float(art_vendas["residual_sigma"])
    sigma_r = float(art_retiradas["residual_sigma"])

    all_rows = []
    summaries_by_scenario = {}

    for scenario_name, sc_cfg in params["forecast"]["scenarios"].items():
        # Estado recursivo inicializado com os valores reais mais recentes
        vendas_buffer = df_closed["vendas"].astype(float).tail(3).tolist()
        vendas_du_last = float(last_closed["vendas_por_du"])
        retiradas_last = float(last_closed["retiradas_sem_anomalia_interp"])
        retiradas_du_last = retiradas_last / float(last_closed["dias_uteis_ponderados"])
        carteira_atual = float(last_closed["carteira_fim_mes"])

        bairros_atual = base_bairros_ativos
        canais_atual = base_canais_ativos

        novos_bairros_m = float(sc_cfg["novos_bairros_ativos_mes"])
        novos_canais_2027 = float(sc_cfg["novos_canais_adicionais"])
        fator_vendas = float(sc_cfg["fator_crescimento_vendas"])
        fator_retiradas = float(sc_cfg["ajuste_retiradas_fator"])

        sc_rows = []
        for step_idx, cal_row in df_future_cal.iterrows():
            anomes = str(cal_row["anomes"])
            ano = int(cal_row["ano"])
            mes = int(cal_row["mes"])
            du = float(cal_row["dias_uteis_ponderados"])
            is_target_year = ano == target_year

            # Evolução de bairros e canais (meses-ponte 2026 iguais entre cenários para partida idêntica em Jan/2027)
            if is_target_year:
                bairros_atual = bairros_atual + novos_bairros_m
                # Se houver novo canal no cenário, entra a partir de Março/2027
                if mes == 3 and novos_canais_2027 > 0:
                    canais_atual = base_canais_ativos + novos_canais_2027
            else:
                bairros_atual = bairros_atual + 2.0

            mes_sin = float(np.sin(2.0 * np.pi * mes / 12.0))
            mes_cos = float(np.cos(2.0 * np.pi * mes / 12.0))

            feat_v = {
                "dias_uteis_ponderados": du,
                "qtd_bairros_ativos_mes": bairros_atual,
                "qtd_canais_ativos_mes": canais_atual,
                "vendas_por_du_lag1": vendas_du_last,
                "vendas_lag1": vendas_buffer[-1],
                "mes_sin": mes_sin,
                "mes_cos": mes_cos,
            }
            vendas_pred = _predict_single_step(art_vendas, feat_v)
            if is_target_year:
                vendas_pred = vendas_pred * fator_vendas

            feat_r = {
                "dias_uteis_ponderados": du,
                "carteira_inicio_mes": carteira_atual,
                "vendas_acum_3m_ant": float(sum(vendas_buffer[-3:])),
                "retiradas_por_du_lag1": retiradas_du_last,
                "retiradas_lag1": retiradas_last,
                "mes_sin": mes_sin,
                "mes_cos": mes_cos,
            }
            retiradas_pred = _predict_single_step(art_retiradas, feat_r)
            if is_target_year:
                retiradas_pred = retiradas_pred * fator_retiradas

            # Intervalo de predição com crescimento suave da incerteza ao longo do horizonte
            h_scale = float(np.sqrt(1.0 + 0.08 * step_idx))
            v_low = max(0.0, vendas_pred - z_ci * sigma_v * h_scale)
            v_high = vendas_pred + z_ci * sigma_v * h_scale
            r_low = max(0.0, retiradas_pred - z_ci * sigma_r * h_scale)
            r_high = retiradas_pred + z_ci * sigma_r * h_scale

            saldo_pred = vendas_pred - retiradas_pred
            carteira_ini = carteira_atual
            carteira_fim = carteira_ini + saldo_pred

            sc_row = {
                "cenario": scenario_name,
                "anomes": anomes,
                "ano": ano,
                "mes": mes,
                "tipo_periodo": "Projecao_2027" if is_target_year else "Ponte_2026",
                "dias_uteis_ponderados": du,
                "qtd_bairros_ativos_proj": round(bairros_atual, 1),
                "qtd_canais_ativos_proj": round(canais_atual, 1),
                "vendas_prev": round(vendas_pred, 1),
                "vendas_ic_inf": round(v_low, 1),
                "vendas_ic_sup": round(v_high, 1),
                "vendas_por_du_prev": round(vendas_pred / du, 2),
                "retiradas_prev": round(retiradas_pred, 1),
                "retiradas_ic_inf": round(r_low, 1),
                "retiradas_ic_sup": round(r_high, 1),
                "retiradas_por_du_prev": round(retiradas_pred / du, 2),
                "saldo_liquido_prev": round(saldo_pred, 1),
                "carteira_inicio_mes_prev": round(carteira_ini, 1),
                "carteira_fim_mes_prev": round(carteira_fim, 1),
                "taxa_churn_carteira_pct": round(retiradas_pred / carteira_ini * 100.0, 2) if carteira_ini > 0 else 0.0,
                "relacao_retirada_venda_pct": round(retiradas_pred / vendas_pred * 100.0, 2) if vendas_pred > 0 else 0.0,
            }
            sc_rows.append(sc_row)
            all_rows.append(sc_row)

            # Atualizar estado recursivo para o próximo mês
            vendas_buffer.append(vendas_pred)
            vendas_du_last = vendas_pred / du
            retiradas_last = retiradas_pred
            retiradas_du_last = retiradas_pred / du
            carteira_atual = carteira_fim

        df_sc_2027 = pd.DataFrame([r for r in sc_rows if r["ano"] == target_year])
        carteira_ini_2027 = float(df_sc_2027.iloc[0]["carteira_inicio_mes_prev"])
        carteira_fim_2027 = float(df_sc_2027.iloc[-1]["carteira_fim_mes_prev"])
        total_vendas_2027 = float(df_sc_2027["vendas_prev"].sum())
        total_retiradas_2027 = float(df_sc_2027["retiradas_prev"].sum())
        saldo_2027 = total_vendas_2027 - total_retiradas_2027
        cresc_pct_2027 = (carteira_fim_2027 / carteira_ini_2027 - 1.0) * 100.0

        summaries_by_scenario[scenario_name] = {
            "descricao": sc_cfg["descricao"],
            "total_dias_uteis_ponderados_2027": round(float(df_sc_2027["dias_uteis_ponderados"].sum()), 1),
            "vendas_total_2027": round(total_vendas_2027, 0),
            "vendas_media_mensal_2027": round(total_vendas_2027 / 12.0, 1),
            "retiradas_total_2027": round(total_retiradas_2027, 0),
            "retiradas_media_mensal_2027": round(total_retiradas_2027 / 12.0, 1),
            "saldo_liquido_2027": round(saldo_2027, 0),
            "carteira_inicio_2027": round(carteira_ini_2027, 0),
            "carteira_fim_2027": round(carteira_fim_2027, 0),
            "crescimento_carteira_2027_pct": round(cresc_pct_2027, 2),
            "churn_medio_mensal_sobre_carteira_pct": round(float(df_sc_2027["taxa_churn_carteira_pct"].mean()), 2),
            "relacao_retirada_venda_2027_pct": round(total_retiradas_2027 / total_vendas_2027 * 100.0, 2),
        }

    df_forecast = pd.DataFrame(all_rows)
    df_forecast.to_parquet(predictions_dir / "projecao_2027.parquet", index=False)
    df_forecast.to_csv(predictions_dir / "projecao_2027.csv", index=False)

    resumo = {
        "target_year": target_year,
        "last_closed_month_used": last_closed_month,
        "carteira_ultimo_mes_fechado": float(last_closed["carteira_fim_mes"]),
        "confidence_level": conf_level,
        "champion_vendas": art_vendas["model_name"],
        "champion_retiradas": art_retiradas["model_name"],
        "cenarios_2027": summaries_by_scenario,
    }
    with open(predictions_dir / "resumo_2027.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)

    return resumo


if __name__ == "__main__":
    res = forecast_2027()
    print(json.dumps(res, ensure_ascii=False, indent=2))
