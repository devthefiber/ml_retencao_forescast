"""
Construção do calendário operacional de dias úteis ponderados (2024–2027).

Regra de negócio definida:
  - Dia útil (segunda a sexta, não feriado): peso 1.0
  - Sábado (não feriado):                    peso 0.5
  - Domingo ou Feriado (BR + GO + Goiânia):  peso 0.0
"""
from __future__ import annotations

from typing import Any

import holidays
import numpy as np
import pandas as pd


def build_calendar(params: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gera o calendário diário e o resumo mensal com pesos operacionais."""
    cal_cfg = params["calendar"]
    weights = cal_cfg["weights"]

    start_date = pd.Timestamp(cal_cfg["start_date"])
    end_date = pd.Timestamp(cal_cfg["end_date"])
    years = list(range(start_date.year, end_date.year + 1))

    # Feriados nacionais + estaduais de Goiás
    br_go_holidays = holidays.country_holidays(
        cal_cfg.get("country", "BR"),
        subdiv=cal_cfg.get("state", "GO"),
        years=years,
    )

    holiday_map: dict[pd.Timestamp, str] = {
        pd.Timestamp(dt): name for dt, name in br_go_holidays.items()
    }

    # Feriados municipais fixos (Goiânia)
    for item in cal_cfg.get("municipal_fixed_holidays", []):
        mm_dd = item["date"]
        name = item["name"]
        for y in years:
            dt = pd.Timestamp(f"{y}-{mm_dd}")
            holiday_map[dt] = name

    # Datas móveis operacionais (Carnaval, Corpus Christi)
    for item in cal_cfg.get("extra_movable_holidays", []):
        dt = pd.Timestamp(item["date"])
        if start_date <= dt <= end_date:
            holiday_map[dt] = item["name"]

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    df_daily = pd.DataFrame({"data": dates})
    df_daily["anomes"] = df_daily["data"].dt.strftime("%Y-%m")
    df_daily["ano"] = df_daily["data"].dt.year
    df_daily["mes"] = df_daily["data"].dt.month
    df_daily["dia_semana"] = df_daily["data"].dt.dayofweek  # 0=Seg .. 5=Sáb, 6=Dom
    df_daily["nome_dia_semana"] = df_daily["dia_semana"].map(
        {0: "Segunda", 1: "Terça", 2: "Quarta", 3: "Quinta", 4: "Sexta", 5: "Sábado", 6: "Domingo"}
    )
    df_daily["nome_feriado"] = df_daily["data"].map(holiday_map)
    df_daily["is_feriado"] = df_daily["nome_feriado"].notna()
    df_daily["is_domingo"] = df_daily["dia_semana"] == 6
    df_daily["is_sabado"] = df_daily["dia_semana"] == 5
    df_daily["is_dia_util_cheio"] = (df_daily["dia_semana"] < 5) & (~df_daily["is_feriado"])
    df_daily["is_sabado_util"] = df_daily["is_sabado"] & (~df_daily["is_feriado"])
    df_daily["is_feriado_em_dia_util"] = df_daily["is_feriado"] & (df_daily["dia_semana"] < 5)

    # Aplicar pesos: Feriado ou Domingo = 0; Sábado não feriado = 0.5; Seg-Sex não feriado = 1.0
    conditions = [
        df_daily["is_feriado"],
        df_daily["is_domingo"],
        df_daily["is_sabado"],
    ]
    choices = [
        float(weights["holiday"]),
        float(weights["sunday"]),
        float(weights["saturday"]),
    ]
    df_daily["peso_dia_util"] = np.select(conditions, choices, default=float(weights["weekday"]))

    tipo_dia_cond = [
        df_daily["is_feriado"],
        df_daily["is_domingo"],
        df_daily["is_sabado"],
    ]
    tipo_dia_lbl = ["Feriado", "Domingo", "Sábado"]
    df_daily["tipo_dia"] = np.select(tipo_dia_cond, tipo_dia_lbl, default="Dia Útil")

    # Resumo mensal
    df_monthly = (
        df_daily.groupby(["anomes", "ano", "mes"], as_index=False)
        .agg(
            dias_calendario=("data", "count"),
            dias_uteis_cheios=("is_dia_util_cheio", "sum"),
            sabados_uteis=("is_sabado_util", "sum"),
            domingos=("is_domingo", "sum"),
            feriados_total=("is_feriado", "sum"),
            feriados_em_dia_util=("is_feriado_em_dia_util", "sum"),
            dias_uteis_ponderados=("peso_dia_util", "sum"),
        )
        .sort_values("anomes")
        .reset_index(drop=True)
    )

    return df_daily, df_monthly


def evaluate_empirical_day_weights(df_daily_series: pd.DataFrame) -> pd.DataFrame:
    """
    Compara empiricamente o volume médio diário de vendas e retiradas por tipo de dia
    (Dia Útil, Sábado, Domingo, Feriado) versus os pesos teóricos (1.0, 0.5, 0.0).
    """
    resumo = (
        df_daily_series.groupby("tipo_dia", as_index=False)
        .agg(
            qtd_dias=("data", "count"),
            media_vendas=("vendas", "mean"),
            media_retiradas=("retiradas", "mean"),
            peso_teorico=("peso_dia_util", "first"),
        )
    )
    base_vendas = resumo.loc[resumo["tipo_dia"] == "Dia Útil", "media_vendas"].values[0]
    base_retiradas = resumo.loc[resumo["tipo_dia"] == "Dia Útil", "media_retiradas"].values[0]

    resumo["peso_empirico_vendas"] = (
        resumo["media_vendas"] / base_vendas if base_vendas > 0 else np.nan
    )
    resumo["peso_empirico_retiradas"] = (
        resumo["media_retiradas"] / base_retiradas if base_retiradas > 0 else np.nan
    )
    ordem = {"Dia Útil": 0, "Sábado": 1, "Domingo": 2, "Feriado": 3}
    resumo["_ord"] = resumo["tipo_dia"].map(ordem)
    return resumo.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)
