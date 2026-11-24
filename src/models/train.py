"""
Treinamento, seleção e diagnóstico de modelos de Regressão Linear (MLOps).

Treina e compara modelos da família de Regressão Linear (OLS, Ridge, Lasso,
ElasticNet e HuberRegressor) para dois alvos principais:
  1. Vendas (Ativações mensais)
  2. Retiradas (Cancelamentos totais do mês, independente da safra)

Aplica validação temporal estrita:
  - Expanding-window TimeSeriesSplit no conjunto de treino (seleção de hiperparâmetros)
  - Holdout out-of-time nos últimos meses fechados de 2026 (avaliação justa sem vazamento)
  - Diagnóstico clássico de regressão linear (VIF, Durbin-Watson, Shapiro-Wilk nos resíduos)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from src.config import DATA_PROCESSED_DIR, MODELS_DIR, ensure_directories, load_params

# Conjuntos de features causais e simuláveis para 2027
VENDAS_FEATURES_TAXA_DU = [
    "qtd_bairros_ativos_mes",
    "qtd_canais_ativos_mes",
    "vendas_por_du_lag1",
    "mes_sin",
    "mes_cos",
]

VENDAS_FEATURES_DIRETO = [
    "dias_uteis_ponderados",
    "qtd_bairros_ativos_mes",
    "qtd_canais_ativos_mes",
    "vendas_lag1",
    "mes_sin",
    "mes_cos",
]

RETIRADAS_FEATURES_TAXA_DU = [
    "carteira_inicio_mes",
    "vendas_acum_3m_ant",
    "retiradas_por_du_lag1",
    "mes_sin",
    "mes_cos",
]

RETIRADAS_FEATURES_DIRETO = [
    "dias_uteis_ponderados",
    "carteira_inicio_mes",
    "vendas_acum_3m_ant",
    "retiradas_lag1",
    "mes_sin",
    "mes_cos",
]


def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 and np.var(y_true) > 0 else float("nan")
    bias = float(np.mean(y_pred - y_true))
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE_pct": round(mape, 2), "R2": round(r2, 4), "Bias": round(bias, 2)}


def _compute_vif(df_X: pd.DataFrame) -> list[dict[str, Any]]:
    X = df_X.astype(float).copy()
    X_const = np.column_stack([np.ones(len(X)), X.values])
    out = []
    for i, col in enumerate(X.columns):
        try:
            vif_val = float(variance_inflation_factor(X_const, i + 1))
        except Exception:
            vif_val = float("nan")
        out.append({"feature": col, "vif": round(vif_val, 3) if np.isfinite(vif_val) else None})
    return out


def _residual_diagnostics(residuals: np.ndarray) -> dict[str, Any]:
    res = np.asarray(residuals, dtype=float)
    dw = float(durbin_watson(res)) if len(res) >= 3 else float("nan")
    if len(res) >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(res)
    else:
        shapiro_stat, shapiro_p = float("nan"), float("nan")
    return {
        "n_obs": int(len(res)),
        "residual_mean": round(float(np.mean(res)), 3),
        "residual_std": round(float(np.std(res, ddof=1)), 3) if len(res) > 1 else 0.0,
        "durbin_watson": round(dw, 3) if np.isfinite(dw) else None,
        "shapiro_stat": round(float(shapiro_stat), 4) if np.isfinite(shapiro_stat) else None,
        "shapiro_pvalue": round(float(shapiro_p), 4) if np.isfinite(shapiro_p) else None,
    }


def _extract_coefficients(pipe: Pipeline, feature_names: list[str]) -> dict[str, Any]:
    """Extrai coeficientes padronizados e na escala original das variáveis."""
    scaler: StandardScaler = pipe.named_steps["scaler"]
    reg = pipe.named_steps["regressor"]
    coef_scaled = np.asarray(reg.coef_, dtype=float)
    coef_orig = coef_scaled / scaler.scale_
    intercept_orig = float(reg.intercept_ - np.sum(coef_scaled * scaler.mean_ / scaler.scale_))
    return {
        "intercept_original": round(intercept_orig, 6),
        "intercept_scaled": round(float(reg.intercept_), 6),
        "features": [
            {
                "feature": f,
                "coef_scaled": round(float(cs), 6),
                "coef_original": round(float(co), 6),
            }
            for f, cs, co in zip(feature_names, coef_scaled, coef_orig)
        ],
    }


def _tune_and_evaluate_candidate(
    name: str,
    estimator_grid: list[tuple[str, Any]],
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    scale_by_du: bool,
    volume_col: str,
    tscv_splits: int,
) -> dict[str, Any]:
    """Seleciona hiperparâmetro via TimeSeriesSplit CV e avalia no holdout temporal."""
    X_train = df_train[feature_cols].astype(float)
    y_train = df_train[target_col].astype(float).values
    du_train = df_train["dias_uteis_ponderados"].astype(float).values
    vol_train = df_train[volume_col].astype(float).values

    X_val = df_val[feature_cols].astype(float)
    du_val = df_val["dias_uteis_ponderados"].astype(float).values
    vol_val = df_val[volume_col].astype(float).values

    n_splits = min(tscv_splits, max(2, len(df_train) // 5))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    best_cv_rmse = np.inf
    best_cfg_name = ""
    best_estimator = None
    best_cv_mape = np.inf

    for cfg_name, est in estimator_grid:
        fold_rmses = []
        fold_mapes = []
        for tr_idx, te_idx in tscv.split(X_train):
            pipe = Pipeline([("scaler", StandardScaler()), ("regressor", est)])
            pipe.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            pred_raw = pipe.predict(X_train.iloc[te_idx])
            pred_vol = pred_raw * du_train[te_idx] if scale_by_du else pred_raw
            pred_vol = np.clip(pred_vol, 0.0, None)
            fold_rmses.append(root_mean_squared_error(vol_train[te_idx], pred_vol))
            fold_mapes.append(mean_absolute_percentage_error(vol_train[te_idx], pred_vol) * 100.0)
        mean_rmse = float(np.mean(fold_rmses))
        mean_mape = float(np.mean(fold_mapes))
        if mean_rmse < best_cv_rmse:
            best_cv_rmse = mean_rmse
            best_cv_mape = mean_mape
            best_cfg_name = cfg_name
            best_estimator = est

    # Treina o melhor hiperparâmetro em todo o bloco de treino e testa no holdout 2026
    pipe_val = Pipeline([("scaler", StandardScaler()), ("regressor", best_estimator)])
    pipe_val.fit(X_train, y_train)

    pred_train_raw = pipe_val.predict(X_train)
    pred_train_vol = np.clip(pred_train_raw * du_train if scale_by_du else pred_train_raw, 0.0, None)
    pred_val_raw = pipe_val.predict(X_val)
    pred_val_vol = np.clip(pred_val_raw * du_val if scale_by_du else pred_val_raw, 0.0, None)

    metrics_train = _calc_metrics(vol_train, pred_train_vol)
    metrics_val = _calc_metrics(vol_val, pred_val_vol)

    # Score combinado (prioriza precisão no holdout recente de 2026 equilibrada com estabilidade CV temporal)
    selection_score = 0.75 * metrics_val["MAPE_pct"] + 0.25 * best_cv_mape

    return {
        "model_name": name,
        "best_config": best_cfg_name,
        "estimator_prototype": best_estimator,
        "target_col": target_col,
        "scale_by_du": scale_by_du,
        "features": feature_cols,
        "cv_rmse": round(best_cv_rmse, 2),
        "cv_mape_pct": round(best_cv_mape, 2),
        "train_metrics": metrics_train,
        "val_metrics": metrics_val,
        "selection_score": round(selection_score, 3),
        "val_predictions": [
            {
                "anomes": m,
                "real": float(r),
                "previsto": round(float(p), 2),
                "erro": round(float(p - r), 2),
                "erro_pct": round(float((p - r) / r * 100.0), 2) if r > 0 else 0.0,
            }
            for m, r, p in zip(df_val["anomes"].tolist(), vol_val, pred_val_vol)
        ],
        "val_residuals": (vol_val - pred_val_vol).tolist(),
        "train_residuals": (vol_train - pred_train_vol).tolist(),
    }


def _build_linear_grids(params: dict[str, Any]) -> dict[str, list[tuple[str, Any]]]:
    m_cfg = params["modeling"]
    return {
        "OLS": [("OLS", LinearRegression())],
        "Ridge": [(f"Ridge(alpha={a})", Ridge(alpha=a)) for a in m_cfg["ridge_alphas"]],
        "Lasso": [(f"Lasso(alpha={a})", Lasso(alpha=a, max_iter=10000)) for a in m_cfg["lasso_alphas"]],
        "ElasticNet": [
            (f"ElasticNet(a={a},l1={l1})", ElasticNet(alpha=a, l1_ratio=l1, max_iter=10000))
            for a in m_cfg["elasticnet_alphas"]
            for l1 in m_cfg["elasticnet_l1_ratios"]
        ],
        "Huber": [
            (f"Huber(eps={e},a={a})", HuberRegressor(epsilon=e, alpha=a, max_iter=5000))
            for e in (1.15, 1.35, 1.6)
            for a in (0.001, 0.01, 0.1, 1.0)
        ],
    }


def _fit_final_production_model(
    champion: dict[str, Any],
    df_full_closed: pd.DataFrame,
    volume_col: str,
) -> tuple[Pipeline, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """
    Re-treina o modelo campeão em 100% dos meses fechados válidos antes de projetar 2027,
    garantindo que os meses mais recentes de 2026 estejam incorporados aos coeficientes.
    """
    from sklearn.base import clone

    X_full = df_full_closed[champion["features"]].astype(float)
    y_full = df_full_closed[champion["target_col"]].astype(float).values
    du_full = df_full_closed["dias_uteis_ponderados"].astype(float).values
    vol_full = df_full_closed[volume_col].astype(float).values

    final_pipe = Pipeline([("scaler", StandardScaler()), ("regressor", clone(champion["estimator_prototype"]))])
    final_pipe.fit(X_full, y_full)

    pred_raw = final_pipe.predict(X_full)
    pred_vol = np.clip(pred_raw * du_full if champion["scale_by_du"] else pred_raw, 0.0, None)
    residuals = vol_full - pred_vol

    full_metrics = _calc_metrics(vol_full, pred_vol)
    diag = _residual_diagnostics(residuals)
    coefs = _extract_coefficients(final_pipe, champion["features"])
    vifs = _compute_vif(X_full)

    return final_pipe, {"full_fit_metrics": full_metrics, "diagnostics": diag, "coefficients": coefs}, diag, vifs


def train_models(
    processed_dir: Path = DATA_PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Executa o treinamento completo, benchmark e persistência dos artefatos de MLOps."""
    ensure_directories()
    params = params or load_params()
    val_months = int(params["modeling"]["validation_months"])
    tscv_splits = int(params["modeling"]["tscv_splits"])

    df_all = pd.read_parquet(processed_dir / "serie_mensal_ml.parquet")
    df_closed = df_all[df_all["mes_fechado"]].sort_values("anomes").reset_index(drop=True)

    # Remover primeiros meses sem lags completos (lag 1..3)
    df_vendas_base = df_closed.dropna(subset=VENDAS_FEATURES_TAXA_DU + VENDAS_FEATURES_DIRETO).reset_index(drop=True)
    df_ret_base_all = df_closed.dropna(subset=RETIRADAS_FEATURES_TAXA_DU + RETIRADAS_FEATURES_DIRETO).reset_index(drop=True)

    # Split temporal estrito (holdout = últimos N meses fechados de 2026)
    val_cutoff_months = df_closed["anomes"].tail(val_months).tolist()

    df_v_train = df_vendas_base[~df_vendas_base["anomes"].isin(val_cutoff_months)].reset_index(drop=True)
    df_v_val = df_vendas_base[df_vendas_base["anomes"].isin(val_cutoff_months)].reset_index(drop=True)

    # Para retiradas: treino limpo exclui picos de baixa em lote; também rodamos benchmark sem filtro
    df_r_train_clean = df_ret_base_all[
        (~df_ret_base_all["anomes"].isin(val_cutoff_months)) & (df_ret_base_all["anomalia_retirada"] == 0)
    ].reset_index(drop=True)
    df_r_train_raw = df_ret_base_all[~df_ret_base_all["anomes"].isin(val_cutoff_months)].reset_index(drop=True)
    df_r_val = df_ret_base_all[df_ret_base_all["anomes"].isin(val_cutoff_months)].reset_index(drop=True)
    df_r_full_clean = df_ret_base_all[df_ret_base_all["anomalia_retirada"] == 0].reset_index(drop=True)

    grids = _build_linear_grids(params)

    # --- 1. Benchmark de Modelos para VENDAS ---
    vendas_experiments = []
    for family in ("OLS", "Ridge", "Lasso", "ElasticNet"):
        # Formulação A: Taxa por Dia Útil Ponderado (1 / 0,5 / 0)
        vendas_experiments.append(
            _tune_and_evaluate_candidate(
                name=f"{family}_Taxa_DU",
                estimator_grid=grids[family],
                df_train=df_v_train,
                df_val=df_v_val,
                feature_cols=VENDAS_FEATURES_TAXA_DU,
                target_col="vendas_por_du",
                scale_by_du=True,
                volume_col="vendas",
                tscv_splits=tscv_splits,
            )
        )
        # Formulação B: Volume Direto com dias_uteis_ponderados como feature
        vendas_experiments.append(
            _tune_and_evaluate_candidate(
                name=f"{family}_Direto",
                estimator_grid=grids[family],
                df_train=df_v_train,
                df_val=df_v_val,
                feature_cols=VENDAS_FEATURES_DIRETO,
                target_col="vendas",
                scale_by_du=False,
                volume_col="vendas",
                tscv_splits=tscv_splits,
            )
        )

    vendas_experiments.sort(key=lambda x: x["selection_score"])
    champ_vendas = vendas_experiments[0]

    # --- 2. Benchmark de Modelos para RETIRADAS (Cancelamentos totais do mês) ---
    retiradas_experiments = []
    for family in ("OLS", "Ridge", "Lasso", "ElasticNet"):
        retiradas_experiments.append(
            _tune_and_evaluate_candidate(
                name=f"{family}_Taxa_DU_Limpo",
                estimator_grid=grids[family],
                df_train=df_r_train_clean,
                df_val=df_r_val,
                feature_cols=RETIRADAS_FEATURES_TAXA_DU,
                target_col="retiradas_por_du",
                scale_by_du=True,
                volume_col="retiradas",
                tscv_splits=tscv_splits,
            )
        )
        retiradas_experiments.append(
            _tune_and_evaluate_candidate(
                name=f"{family}_Direto_Limpo",
                estimator_grid=grids[family],
                df_train=df_r_train_clean,
                df_val=df_r_val,
                feature_cols=RETIRADAS_FEATURES_DIRETO,
                target_col="retiradas",
                scale_by_du=False,
                volume_col="retiradas",
                tscv_splits=tscv_splits,
            )
        )

    # Benchmarks extras com TODAS as observações (incluindo baixas em lote 2025-06/2025-08)
    retiradas_experiments.append(
        _tune_and_evaluate_candidate(
            name="Huber_Taxa_DU_ComAnomalias",
            estimator_grid=grids["Huber"],
            df_train=df_r_train_raw,
            df_val=df_r_val,
            feature_cols=RETIRADAS_FEATURES_TAXA_DU,
            target_col="retiradas_por_du",
            scale_by_du=True,
            volume_col="retiradas",
            tscv_splits=tscv_splits,
        )
    )
    retiradas_experiments.append(
        _tune_and_evaluate_candidate(
            name="OLS_Direto_SemTratarAnomalias",
            estimator_grid=grids["OLS"],
            df_train=df_r_train_raw,
            df_val=df_r_val,
            feature_cols=RETIRADAS_FEATURES_DIRETO,
            target_col="retiradas",
            scale_by_du=False,
            volume_col="retiradas",
            tscv_splits=tscv_splits,
        )
    )

    retiradas_experiments.sort(key=lambda x: x["selection_score"])
    champ_retiradas = retiradas_experiments[0]

    # --- 3. Re-treinar campeões com 100% dos meses fechados para produção 2027 ---
    pipe_vendas_prod, vendas_prod_info, _, vendas_vifs = _fit_final_production_model(
        champ_vendas, df_vendas_base, "vendas"
    )
    df_ret_prod_fit = (
        df_ret_base_all if "ComAnomalias" in champ_retiradas["model_name"] or "SemTratar" in champ_retiradas["model_name"] else df_r_full_clean
    )
    pipe_retiradas_prod, ret_prod_info, _, ret_vifs = _fit_final_production_model(
        champ_retiradas, df_ret_prod_fit, "retiradas"
    )

    # Desvio padrão residual fora da amostra (holdout + full) para intervalo de predição 2027
    sigma_vendas = float(max(champ_vendas["val_metrics"]["RMSE"], vendas_prod_info["diagnostics"]["residual_std"]))
    sigma_retiradas = float(max(champ_retiradas["val_metrics"]["RMSE"], ret_prod_info["diagnostics"]["residual_std"]))

    # Persistir artefatos de modelo
    joblib.dump(
        {
            "pipeline": pipe_vendas_prod,
            "model_name": champ_vendas["model_name"],
            "best_config": champ_vendas["best_config"],
            "features": champ_vendas["features"],
            "target_col": champ_vendas["target_col"],
            "scale_by_du": champ_vendas["scale_by_du"],
            "residual_sigma": sigma_vendas,
        },
        models_dir / "modelo_vendas.joblib",
    )

    joblib.dump(
        {
            "pipeline": pipe_retiradas_prod,
            "model_name": champ_retiradas["model_name"],
            "best_config": champ_retiradas["best_config"],
            "features": champ_retiradas["features"],
            "target_col": champ_retiradas["target_col"],
            "scale_by_du": champ_retiradas["scale_by_du"],
            "residual_sigma": sigma_retiradas,
        },
        models_dir / "modelo_retiradas.joblib",
    )

    def _strip_proto(exp_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in e.items() if k != "estimator_prototype"} for e in exp_list]

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_version": params["project"]["version"],
        "train_months_vendas": [df_v_train["anomes"].min(), df_v_train["anomes"].max()],
        "validation_months": val_cutoff_months,
        "anomalias_excluidas_treino_retiradas": df_ret_base_all.loc[
            df_ret_base_all["anomalia_retirada"] == 1, "anomes"
        ].tolist(),
        "champion_vendas": {
            "model_name": champ_vendas["model_name"],
            "best_config": champ_vendas["best_config"],
            "features": champ_vendas["features"],
            "scale_by_du": champ_vendas["scale_by_du"],
            "cv_mape_pct": champ_vendas["cv_mape_pct"],
            "val_metrics": champ_vendas["val_metrics"],
            "val_predictions": champ_vendas["val_predictions"],
            "production_fit": vendas_prod_info,
            "vif": vendas_vifs,
            "residual_sigma_forecast": round(sigma_vendas, 2),
        },
        "champion_retiradas": {
            "model_name": champ_retiradas["model_name"],
            "best_config": champ_retiradas["best_config"],
            "features": champ_retiradas["features"],
            "scale_by_du": champ_retiradas["scale_by_du"],
            "cv_mape_pct": champ_retiradas["cv_mape_pct"],
            "val_metrics": champ_retiradas["val_metrics"],
            "val_predictions": champ_retiradas["val_predictions"],
            "production_fit": ret_prod_info,
            "vif": ret_vifs,
            "residual_sigma_forecast": round(sigma_retiradas, 2),
        },
        "experiments_vendas": _strip_proto(vendas_experiments),
        "experiments_retiradas": _strip_proto(retiradas_experiments),
    }

    with open(models_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Salvar tabelas resumidas de benchmark para os notebooks
    pd.DataFrame(
        [
            {
                "modelo": e["model_name"],
                "config": e["best_config"],
                "usa_taxa_du": e["scale_by_du"],
                "cv_rmse": e["cv_rmse"],
                "cv_mape_pct": e["cv_mape_pct"],
                "val_mae": e["val_metrics"]["MAE"],
                "val_rmse": e["val_metrics"]["RMSE"],
                "val_mape_pct": e["val_metrics"]["MAPE_pct"],
                "val_r2": e["val_metrics"]["R2"],
                "score": e["selection_score"],
            }
            for e in vendas_experiments
        ]
    ).to_csv(models_dir / "benchmark_vendas.csv", index=False)

    pd.DataFrame(
        [
            {
                "modelo": e["model_name"],
                "config": e["best_config"],
                "usa_taxa_du": e["scale_by_du"],
                "cv_rmse": e["cv_rmse"],
                "cv_mape_pct": e["cv_mape_pct"],
                "val_mae": e["val_metrics"]["MAE"],
                "val_rmse": e["val_metrics"]["RMSE"],
                "val_mape_pct": e["val_metrics"]["MAPE_pct"],
                "val_r2": e["val_metrics"]["R2"],
                "score": e["selection_score"],
            }
            for e in retiradas_experiments
        ]
    ).to_csv(models_dir / "benchmark_retiradas.csv", index=False)

    return metadata


if __name__ == "__main__":
    meta = train_models()
    print("Campeão Vendas:", meta["champion_vendas"]["model_name"], meta["champion_vendas"]["val_metrics"])
    print("Campeão Retiradas:", meta["champion_retiradas"]["model_name"], meta["champion_retiradas"]["val_metrics"])
