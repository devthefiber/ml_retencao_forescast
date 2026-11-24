"""Carregamento centralizado de configurações, caminhos e credenciais de banco."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "params.yaml"
DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"


def ensure_directories() -> None:
    """Garante que todas as pastas de artefatos do pipeline existam."""
    for d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_PREDICTIONS_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_params(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Lê o arquivo configs/params.yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_db_config() -> dict[str, str]:
    """
    Carrega credenciais de leitura do PostgreSQL DW.
    Prioriza o .env local do projeto e faz fallback para o .env do dash-retencao.
    """
    local_env = ROOT_DIR / ".env"
    if local_env.exists():
        load_dotenv(local_env, override=True)

    fallback_env = Path(
        os.getenv("DASH_RETENCAO_ENV", "/home/desenv/projetos/dash-retencao/.env")
    )
    if not os.getenv("DB_NAME") and fallback_env.exists():
        load_dotenv(fallback_env, override=False)

    required = ["DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT"]
    cfg = {k: (os.getenv(k) or "").strip() for k in required}
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise EnvironmentError(
            f"Variáveis de banco ausentes: {missing}. Configure .env ou DASH_RETENCAO_ENV."
        )
    return cfg
