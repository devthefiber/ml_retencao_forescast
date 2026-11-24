"""
Ponto de entrada CLI do pipeline MLOps (Extract -> Features -> Train -> Predict).

Uso:
    python -m src.pipeline --step all
    python -m src.pipeline --step features
    python -m src.pipeline --step train
    python -m src.pipeline --step predict
"""
from __future__ import annotations

import argparse
import json
import sys

from src.data.extract import extract_from_dw
from src.features.build import build_features
from src.models.predict import forecast_2027
from src.models.train import train_models


def run_pipeline(step: str = "all") -> dict:
    results = {}
    if step in ("all", "extract"):
        print("[1/4] Extraindo dados do Data Warehouse (PostgreSQL)...")
        results["extract"] = extract_from_dw()

    if step in ("all", "features"):
        print("[2/4] Construindo calendário ponderado (1 / 0,5 / 0) e features analíticas...")
        results["features"] = build_features()

    if step in ("all", "train"):
        print("[3/4] Treinando e validando modelos de Regressão Linear (backtesting temporal)...")
        results["train"] = train_models()

    if step in ("all", "predict"):
        print("[4/4] Gerando projeção mensal em cascata para 2027 (Conservador / Base / Expansão)...")
        results["predict"] = forecast_2027()

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline MLOps - ML Retenção Forecast 2027")
    parser.add_argument(
        "--step",
        choices=["all", "extract", "features", "train", "predict"],
        default="all",
        help="Etapa do pipeline a executar (padrão: all).",
    )
    args = parser.parse_args()
    res = run_pipeline(args.step)

    if "predict" in res:
        print("\n=== RESUMO EXECUTIVO PROJEÇÃO 2027 ===")
        print(json.dumps(res["predict"]["cenarios_2027"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
