"""
Extração somente leitura do Data Warehouse (PostgreSQL - db_thefiber).

Replica fielmente as regras de negócio e De-Para do projeto `dash-retencao`
(desduplicação por id_contrato, mapeamento de vendedores com mes_transferencia,
normalização de cidades/redes e filtragem de motivos/vendedores ignorados).
"""
from __future__ import annotations

import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg2

from src.config import DATA_RAW_DIR, ensure_directories, load_db_config

# Constantes espelhadas de dash-retencao/dashboard/services/churn_service.py
CIDADES_PRINCIPAIS = [
    "Goiânia",
    "Trindade",
    "Aparecida de Goiânia",
    "Abadia de Goiás",
    "Goianira",
]

BAIRROS_TRINDADE = (
    "PAI ETERNO",
    "NOVA CANAA",
    "SOLAR EMBAUBA",
    "JARDIM SCALA",
    "SETOR DOS BANDEIRANTES",
    "SOL DOURADO",
    "RESIDENCIAL SANTA FE",
    "SETOR MAYSA",
    "RENATA PARK",
    "PONTAKAYANA",
    "SAMORAH",
    "JARDIM IMPERIAL",
    "JARDIM MARISTA",
    "SETOR SUL",
    "VILA PERPETUO SOCORRO",
)
BAIRROS_ABADIA = ("MARIA DE OLIVEIRA", "MARIA OLIVEIRA")
BAIRROS_GOIANIRA = (
    "SOLAR DAS PAINEIRAS",
    "RECANTO SONHADO",
    "JARDIM DAS MARGARIDAS",
    "SETOR MARIAPOLIS",
    "RESIDENCIAL SAO BERNARDO",
    "FONTE DAS AGUAS",
)


def normalize_cidade(cidade_raw: Any, bairro_raw: Any = None) -> str:
    """Normaliza o nome da cidade usando a mesma regra do dash-retencao."""
    b_ascii = ""
    if bairro_raw is not None and pd.notna(bairro_raw):
        b_ascii = (
            unicodedata.normalize("NFKD", str(bairro_raw))
            .encode("ASCII", "ignore")
            .decode("ASCII")
            .upper()
            .strip()
        )

    if cidade_raw is not None and pd.notna(cidade_raw):
        c = str(cidade_raw).strip()
        if c and c.lower() != "none":
            c_ascii = (
                unicodedata.normalize("NFKD", c)
                .encode("ASCII", "ignore")
                .decode("ASCII")
                .upper()
                .strip()
            )
            if c_ascii == "GOIANIA":
                return "Goiânia"
            if c_ascii == "TRINDADE":
                return "Trindade"
            if c_ascii == "APARECIDA DE GOIANIA":
                return "Aparecida de Goiânia"
            if c_ascii == "ABADIA DE GOIAS":
                if b_ascii and any(x in b_ascii for x in ("SANTA FE I", "FORTEVILLE")):
                    return "Goiânia"
                return "Abadia de Goiás"
            if c_ascii == "GOIANIRA":
                if b_ascii and any(x in b_ascii for x in ("CHACARAS SANTA RITA", "JARDIM EUROPA")):
                    return "Goiânia"
                return "Goianira"

    if b_ascii:
        if any(bt in b_ascii for bt in BAIRROS_ABADIA):
            return "Abadia de Goiás"
        if any(bg in b_ascii for bg in BAIRROS_GOIANIRA):
            return "Goianira"
        if any(btr in b_ascii for btr in BAIRROS_TRINDADE) and "SANTA FE I" not in b_ascii:
            return "Trindade"

    return "Goiânia"


def _get_connection():
    cfg = load_db_config()
    return psycopg2.connect(
        dbname=cfg["DB_NAME"],
        user=cfg["DB_USER"],
        password=cfg["DB_PASSWORD"],
        host=cfg["DB_HOST"],
        port=cfg["DB_PORT"],
        options="-c default_transaction_read_only=on",
    )


def _query_to_df(conn, sql: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def _clean_date_str(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"None": None, "nan": None, "NaT": None, "": None})
    return s


def extract_from_dw(output_dir: Path = DATA_RAW_DIR) -> dict[str, Any]:
    """
    Extrai `fato_churn_contrato` e tabelas De-Para do PostgreSQL, aplica
    normalizações de negócio e persiste snapshots versionados em Parquet.
    """
    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = _get_connection()
    try:
        df_fato = _query_to_df(
            conn,
            """
            SELECT
                id_contrato,
                id_cliente,
                nome_vendedor,
                bairro_padronizado,
                cidade,
                tipo_rede_bairro,
                plano_de_venda,
                status_contrato,
                flag_ativacao,
                flag_churn,
                anomes_ativacao,
                anomes_cancelamento,
                data_ativacao,
                data_cancelamento,
                canal_venda_ativacao,
                canal_venda_cancelamento,
                agrupador_motivo,
                descricao_motivo,
                fpd_data_vencimento,
                fpd_status_fatura,
                fpd_dias_atraso,
                fpd_classificacao,
                flag_fpd_61d,
                qtd_os_fpd
            FROM fato_churn_contrato
            """,
        )

        df_canais = _query_to_df(
            conn, "SELECT id, nome, is_active FROM dashboard_canalvenda"
        )
        df_funcs = _query_to_df(
            conn,
            """
            SELECT
                f.ixc_id,
                f.nome,
                f.status_mapeamento,
                f.mes_transferencia,
                f.ignorar_metricas,
                c1.nome AS canal_padrao,
                c2.nome AS canal_transferencia
            FROM dashboard_funcionarioixc f
            LEFT JOIN dashboard_canalvenda c1 ON f.canal_venda_id = c1.id
            LEFT JOIN dashboard_canalvenda c2 ON f.canal_venda_transferencia_id = c2.id
            """,
        )
        df_bairros = _query_to_df(
            conn,
            "SELECT cidade, bairro_ixc, bairro_api, tipo_rede, is_active FROM dashboard_bairrorede",
        )
        df_motivos = _query_to_df(
            conn,
            "SELECT descricao_motivo, agrupamento, ignorar_metricas, is_active FROM dashboard_motivocancelamentoixc",
        )
    finally:
        conn.close()

    total_bruto = len(df_fato)

    # 1. Desduplicar por id_contrato (mesma lógica seen_contratos do dash-retencao)
    df_fato["id_contrato"] = df_fato["id_contrato"].astype(str).str.strip()
    df_fato = df_fato.drop_duplicates(subset=["id_contrato"], keep="first").reset_index(drop=True)

    # Limpar campos de texto/data
    for col in ("anomes_ativacao", "anomes_cancelamento", "data_ativacao", "data_cancelamento", "fpd_data_vencimento"):
        df_fato[col] = _clean_date_str(df_fato[col])

    # 2. Dicionários De-Para em memória
    func_map: dict[str, dict[str, Any]] = {}
    for _, r in df_funcs.iterrows():
        key = str(r["nome"]).strip().upper() if pd.notna(r["nome"]) else ""
        if not key:
            continue
        func_map[key] = {
            "canal_padrao": r["canal_padrao"] if pd.notna(r["canal_padrao"]) and str(r["canal_padrao"]).strip() else "Não vinculado",
            "mes_transferencia": str(r["mes_transferencia"]).strip() if pd.notna(r["mes_transferencia"]) and str(r["mes_transferencia"]).strip() else None,
            "canal_transferencia": r["canal_transferencia"] if pd.notna(r["canal_transferencia"]) and str(r["canal_transferencia"]).strip() else None,
            "ignorar_metricas": bool(r["ignorar_metricas"]),
        }

    motivo_map: dict[str, dict[str, Any]] = {}
    for _, r in df_motivos.iterrows():
        key = str(r["descricao_motivo"]) if pd.notna(r["descricao_motivo"]) else ""
        motivo_map[key] = {
            "agrupamento": r["agrupamento"] if pd.notna(r["agrupamento"]) else "Não mapeado",
            "ignorar_metricas": bool(r["ignorar_metricas"]),
        }

    bairro_map: dict[str, str] = {}
    for _, r in df_bairros.iterrows():
        b_api = str(r["bairro_api"]).upper().strip() if pd.notna(r["bairro_api"]) else ""
        c_norm = normalize_cidade(r["cidade"], r["bairro_api"]).upper().strip()
        c_raw = str(r["cidade"]).upper().strip() if pd.notna(r["cidade"]) else c_norm
        t_rede = str(r["tipo_rede"]).strip() if pd.notna(r["tipo_rede"]) else "Não mapeado"
        if b_api:
            bairro_map[f"{b_api}-{c_norm}"] = t_rede
            bairro_map[f"{b_api}-{c_raw}"] = t_rede

    # 3. Enriquecer cada contrato com as regras exatas do dash-retencao
    cidades_norm = []
    bairros_norm = []
    redes_efetivas = []
    canais_efetivos = []
    ignorar_vendedor = []
    is_vendedor_padrao = []
    agrupamentos_motivo = []
    ignorar_motivo = []

    for row in df_fato.itertuples(index=False):
        cid_norm = normalize_cidade(row.cidade, row.bairro_padronizado)
        cidades_norm.append(cid_norm)

        b_str = str(row.bairro_padronizado).strip().upper() if pd.notna(row.bairro_padronizado) and str(row.bairro_padronizado).strip() else "NÃO INFORMADO"
        bairros_norm.append(b_str)

        c_raw_u = str(row.cidade).upper().strip() if pd.notna(row.cidade) else ""
        override = bairro_map.get(f"{b_str}-{c_raw_u}") or bairro_map.get(f"{b_str}-{cid_norm.upper().strip()}")
        if override and override != "Não mapeado":
            rede = override
        else:
            rede = str(row.tipo_rede_bairro).strip() if pd.notna(row.tipo_rede_bairro) and str(row.tipo_rede_bairro).strip() else "Não mapeado"
        if rede == "Rede Propria":
            rede = "Rede Própria"
        redes_efetivas.append(rede)

        v_norm = str(row.nome_vendedor).strip().upper() if pd.notna(row.nome_vendedor) else ""
        f_info = func_map.get(v_norm, {})
        ign_v = bool(f_info.get("ignorar_metricas", False))
        ignorar_vendedor.append(ign_v)

        canal = f_info.get("canal_padrao", "Não vinculado")
        mes_atv = row.anomes_ativacao
        if f_info.get("mes_transferencia") and mes_atv and mes_atv >= f_info["mes_transferencia"]:
            canal = f_info.get("canal_transferencia") or canal
        if not canal or str(canal).strip() == "":
            canal = "Não vinculado"
        canais_efetivos.append(canal)
        is_vendedor_padrao.append(canal == "Vendedor Padrão")

        desc_m = str(row.descricao_motivo) if pd.notna(row.descricao_motivo) else ""
        m_info = motivo_map.get(desc_m, {"agrupamento": "Não mapeado", "ignorar_metricas": False})
        agrupamentos_motivo.append(m_info["agrupamento"])
        ignorar_motivo.append(bool(m_info["ignorar_metricas"]))

    df_fato["cidade_norm"] = cidades_norm
    df_fato["bairro_norm"] = bairros_norm
    df_fato["tipo_rede_efetivo"] = redes_efetivas
    df_fato["canal_venda_efetivo"] = canais_efetivos
    df_fato["ignorar_vendedor"] = ignorar_vendedor
    df_fato["is_vendedor_padrao"] = is_vendedor_padrao
    df_fato["agrupamento_motivo_efetivo"] = agrupamentos_motivo
    df_fato["ignorar_motivo"] = ignorar_motivo

    # Persistir snapshots em Parquet
    df_fato.to_parquet(output_dir / "contratos_tratados.parquet", index=False)
    df_canais.to_parquet(output_dir / "depara_canais.parquet", index=False)
    df_funcs.to_parquet(output_dir / "depara_funcionarios.parquet", index=False)
    df_bairros.to_parquet(output_dir / "depara_bairros.parquet", index=False)
    df_motivos.to_parquet(output_dir / "depara_motivos.parquet", index=False)

    max_data_atv = df_fato["data_ativacao"].dropna().max()
    max_data_canc = df_fato["data_cancelamento"].dropna().max()

    metadata = {
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_table": "fato_churn_contrato",
        "rows_raw": int(total_bruto),
        "rows_deduplicated": int(len(df_fato)),
        "duplicates_removed": int(total_bruto - len(df_fato)),
        "min_anomes_ativacao": str(df_fato["anomes_ativacao"].dropna().min()),
        "max_anomes_ativacao": str(df_fato["anomes_ativacao"].dropna().max()),
        "max_data_ativacao": str(max_data_atv),
        "max_data_cancelamento": str(max_data_canc),
        "depara_counts": {
            "canais": int(len(df_canais)),
            "funcionarios": int(len(df_funcs)),
            "bairros": int(len(df_bairros)),
            "motivos": int(len(df_motivos)),
        },
    }
    with open(output_dir / "extract_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


if __name__ == "__main__":
    meta = extract_from_dw()
    print(json.dumps(meta, ensure_ascii=False, indent=2))
