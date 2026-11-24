"""Testes automatizados de qualidade de dados, calendário ponderado e artefatos MLOps."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATA_PREDICTIONS_DIR, DATA_PROCESSED_DIR, MODELS_DIR, load_params
from src.features.calendar import build_calendar


def test_calendar_weights_business_rule():
    params = load_params()
    df_daily, df_monthly = build_calendar(params)

    # Pesos permitidos: apenas 0.0, 0.5 e 1.0
    assert set(df_daily["peso_dia_util"].unique()).issubset({0.0, 0.5, 1.0})

    # Todo domingo e todo feriado deve ter peso 0.0
    assert (df_daily.loc[df_daily["is_domingo"], "peso_dia_util"] == 0.0).all()
    assert (df_daily.loc[df_daily["is_feriado"], "peso_dia_util"] == 0.0).all()

    # Todo sábado não feriado deve ter peso 0.5
    sat_mask = df_daily["is_sabado"] & (~df_daily["is_feriado"])
    assert (df_daily.loc[sat_mask, "peso_dia_util"] == 0.5).all()

    # Todo dia útil (seg-sex) não feriado deve ter peso 1.0
    wd_mask = (df_daily["dia_semana"] < 5) & (~df_daily["is_feriado"])
    assert (df_daily.loc[wd_mask, "peso_dia_util"] == 1.0).all()

    # 2027 deve ter exatamente 12 meses completos no calendário
    assert len(df_monthly[df_monthly["ano"] == 2027]) == 12
    # Verificar Natal 2027 (Sábado) e Carnaval 2027 (08/02 e 09/02) com peso 0
    for dt_str in ("2027-02-08", "2027-02-09", "2027-10-24", "2027-12-25"):
        row = df_daily[df_daily["data"] == pd.Timestamp(dt_str)]
        assert len(row) == 1
        assert float(row["peso_dia_util"].iloc[0]) == 0.0


def test_processed_series_integrity():
    path = DATA_PROCESSED_DIR / "serie_mensal_ml.parquet"
    assert path.exists(), "Execute `python -m src.pipeline --step all` antes dos testes."
    df = pd.read_parquet(path)
    df_closed = df[df["mes_fechado"]]

    assert len(df_closed) >= 24
    assert (df_closed["vendas"] > 0).all()
    assert (df_closed["retiradas"] > 0).all()
    assert (df_closed["dias_uteis_ponderados"] >= 18.0).all()
    assert (df_closed["carteira_fim_mes"] > 0).all()
    # Verificar identidade contábil: saldo_liquido = vendas - retiradas
    assert (df_closed["saldo_liquido"] == df_closed["vendas"] - df_closed["retiradas"]).all()


def test_models_and_2027_forecast_artifacts():
    assert (MODELS_DIR / "modelo_vendas.joblib").exists()
    assert (MODELS_DIR / "modelo_retiradas.joblib").exists()
    assert (MODELS_DIR / "metadata.json").exists()

    pred_path = DATA_PREDICTIONS_DIR / "projecao_2027.parquet"
    assert pred_path.exists()
    df_pred = pd.read_parquet(pred_path)
    df_2027 = df_pred[df_pred["ano"] == 2027]

    # 3 cenários x 12 meses = 36 linhas para 2027
    assert set(df_2027["cenario"].unique()) == {"Conservador", "Base", "Expansao"}
    assert len(df_2027) == 36
    assert (df_2027["vendas_prev"] > 0).all()
    assert (df_2027["retiradas_prev"] > 0).all()

    # Coerência entre cenários: Expansão deve gerar carteira final maior que Base, e Base maior que Conservador
    fim_dez = df_2027[df_2027["mes"] == 12].set_index("cenario")["carteira_fim_mes_prev"]
    assert fim_dez["Expansao"] > fim_dez["Base"] > fim_dez["Conservador"]
