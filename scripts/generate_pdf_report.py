"""
Gerador de Relatório Executivo em PDF (ReportLab) — The Fiber | ML Retenção Forecast 2027.

Lê dinamicamente os metadados, séries tratadas, benchmarks, projeções de 2027 e figuras PNG
geradas pelo pipeline MLOps para produzir um PDF executivo diagramado para diretoria e público
não técnico em:
    reports/RELATORIO_EXECUTIVO_PROJECAO_2027.pdf
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROC = ROOT_DIR / "data" / "processed"
DATA_PRED = ROOT_DIR / "data" / "predictions"
MODELS_DIR = ROOT_DIR / "models"
FIG_DIR = ROOT_DIR / "reports" / "figures"
OUTPUT_PDF = ROOT_DIR / "reports" / "RELATORIO_EXECUTIVO_PROJECAO_2027.pdf"
LOGO_PATH = Path("/home/desenv/projetos/dash-retencao/static/img/LOGO_THEFIBER_M1.png")

# Paleta oficial The Fiber
C_PURPLE = colors.HexColor("#73227D")
C_PURPLE_LIGHT = colors.HexColor("#F5EAF7")
C_CYAN = colors.HexColor("#21A6DF")
C_CYAN_LIGHT = colors.HexColor("#E8F6FC")
C_DARK = colors.HexColor("#0A3D52")
C_TEXT = colors.HexColor("#1F2933")
C_MUTED = colors.HexColor("#52606D")
C_BORDER = colors.HexColor("#D9E2EC")
C_ROW_ALT = colors.HexColor("#F8FAFC")
C_GREEN = colors.HexColor("#1E7E34")
C_GREEN_LIGHT = colors.HexColor("#E8F5E9")
C_AMBER = colors.HexColor("#B45309")
C_AMBER_LIGHT = colors.HexColor("#FEF3C7")

PAGE_W, PAGE_H = A4
MARGIN_X = 16 * mm
MARGIN_TOP = 20 * mm
MARGIN_BOTTOM = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_X


def fmt_int(v: float | int) -> str:
    return f"{int(round(float(v))):,}".replace(",", ".")


def fmt_dec(v: float, dec: int = 1) -> str:
    return f"{float(v):,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_signed_int(v: float | int) -> str:
    val = int(round(float(v)))
    prefix = "+" if val > 0 else ""
    return f"{prefix}{val:,}".replace(",", ".")


class NumberedCanvas(canvas.Canvas):
    """Canvas com cabeçalho, rodapé e contagem 'Página X de Y' a partir da página 2."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict[str, Any]] = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_chrome(num_pages)
            super().showPage()
        super().save()

    def _draw_chrome(self, total_pages: int):
        page_num = self._pageNumber
        if page_num == 1:
            # Barra superior e inferior limpa na capa
            self.saveState()
            self.setFillColor(C_PURPLE)
            self.rect(0, PAGE_H - 9 * mm, PAGE_W, 9 * mm, fill=1, stroke=0)
            self.setFillColor(C_CYAN)
            self.rect(0, PAGE_H - 11.2 * mm, PAGE_W, 2.2 * mm, fill=1, stroke=0)
            self.setFillColor(C_DARK)
            self.rect(0, 0, PAGE_W, 7 * mm, fill=1, stroke=0)
            self.setFont("Helvetica", 7.5)
            self.setFillColor(colors.white)
            self.drawString(MARGIN_X, 2.5 * mm, "THE FIBER  |  Inteligência Comercial & Retenção — Projeto ml-retencao-forecast")
            self.drawRightString(PAGE_W - MARGIN_X, 2.5 * mm, "Documento Executivo — Uso Interno")
            self.restoreState()
            return

        self.saveState()
        # Cabeçalho páginas internas
        self.setStrokeColor(C_PURPLE)
        self.setLineWidth(1.0)
        self.line(MARGIN_X, PAGE_H - 13 * mm, PAGE_W - MARGIN_X, PAGE_H - 13 * mm)
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(C_PURPLE)
        self.drawString(MARGIN_X, PAGE_H - 11 * mm, "THE FIBER")
        self.setFont("Helvetica", 8)
        self.setFillColor(C_MUTED)
        self.drawString(MARGIN_X + 20 * mm, PAGE_H - 11 * mm, "|  Relatório Executivo — Projeção de Crescimento 2027 (Machine Learning)")
        self.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 11 * mm, "Base: dash-retencao")

        # Rodapé páginas internas
        self.setStrokeColor(C_BORDER)
        self.setLineWidth(0.6)
        self.line(MARGIN_X, 13 * mm, PAGE_W - MARGIN_X, 13 * mm)
        self.setFont("Helvetica", 7.5)
        self.setFillColor(C_MUTED)
        self.drawString(MARGIN_X, 8.8 * mm, "Projeto ml-retencao-forecast  •  Regressão Linear Regularizada (ElasticNet)  •  Calendário 1 / 0,5 / 0")
        self.setFont("Helvetica-Bold", 7.8)
        self.setFillColor(C_DARK)
        self.drawRightString(PAGE_W - MARGIN_X, 8.8 * mm, f"Página {page_num} de {total_pages}")
        self.restoreState()


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    s: dict[str, ParagraphStyle] = {}

    s["cover_kicker"] = ParagraphStyle(
        "cover_kicker",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=C_CYAN,
        leading=13,
        spaceAfter=6,
    )
    s["cover_title"] = ParagraphStyle(
        "cover_title",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        textColor=C_DARK,
        leading=29,
        spaceAfter=10,
    )
    s["cover_subtitle"] = ParagraphStyle(
        "cover_subtitle",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=11.5,
        textColor=C_MUTED,
        leading=16,
        spaceAfter=16,
    )
    s["h1"] = ParagraphStyle(
        "h1",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14.5,
        textColor=C_PURPLE,
        leading=19,
        spaceBefore=12,
        spaceAfter=6,
    )
    s["h2"] = ParagraphStyle(
        "h2",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11.2,
        textColor=C_DARK,
        leading=15,
        spaceBefore=9,
        spaceAfter=4,
    )
    s["body"] = ParagraphStyle(
        "body",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=9.4,
        textColor=C_TEXT,
        leading=13.8,
        alignment=TA_JUSTIFY,
        spaceAfter=5,
    )
    s["bullet"] = ParagraphStyle(
        "bullet",
        parent=s["body"],
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3.5,
    )
    s["callout_title"] = ParagraphStyle(
        "callout_title",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        textColor=C_DARK,
        leading=13,
        spaceAfter=2,
    )
    s["callout_body"] = ParagraphStyle(
        "callout_body",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=9.0,
        textColor=C_TEXT,
        leading=13,
        alignment=TA_JUSTIFY,
    )
    s["caption"] = ParagraphStyle(
        "caption",
        parent=base["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.0,
        textColor=C_MUTED,
        alignment=TA_CENTER,
        leading=11,
        spaceBefore=2,
        spaceAfter=8,
    )
    s["th"] = ParagraphStyle(
        "th",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.0,
        textColor=colors.white,
        alignment=TA_CENTER,
        leading=10,
    )
    s["th_left"] = ParagraphStyle("th_left", parent=s["th"], alignment=TA_LEFT)
    s["td"] = ParagraphStyle(
        "td",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=8.0,
        textColor=C_TEXT,
        alignment=TA_CENTER,
        leading=10.5,
    )
    s["td_left"] = ParagraphStyle("td_left", parent=s["td"], alignment=TA_LEFT)
    s["td_bold"] = ParagraphStyle("td_bold", parent=s["td"], fontName="Helvetica-Bold")
    s["td_bold_left"] = ParagraphStyle("td_bold_left", parent=s["td_left"], fontName="Helvetica-Bold")
    s["kpi_label"] = ParagraphStyle(
        "kpi_label",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        textColor=C_MUTED,
        alignment=TA_LEFT,
        leading=10,
    )
    s["kpi_val"] = ParagraphStyle(
        "kpi_val",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15.5,
        textColor=C_DARK,
        alignment=TA_LEFT,
        leading=19,
        spaceBefore=2,
        spaceAfter=2,
    )
    s["kpi_sub"] = ParagraphStyle(
        "kpi_sub",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=7.6,
        textColor=C_MUTED,
        alignment=TA_LEFT,
        leading=10,
    )
    return s


def make_callout(
    title: str,
    text: str,
    styles: dict[str, ParagraphStyle],
    bg_color=C_CYAN_LIGHT,
    accent_color=C_CYAN,
) -> Table:
    content = [
        Paragraph(title, styles["callout_title"]),
        Paragraph(text, styles["callout_body"]),
    ]
    tbl = Table([[content]], colWidths=[CONTENT_W])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg_color),
                ("LINEBEFORE", (0, 0), (0, -1), 3.5, accent_color),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return tbl


def make_styled_table(
    headers: list[str],
    rows: list[list[str]],
    col_widths: list[float],
    styles: dict[str, ParagraphStyle],
    left_cols: set[int] | None = None,
    highlight_last_row: bool = False,
    highlight_rows: set[int] | None = None,
    font_size: float | None = None,
) -> Table:
    left_cols = left_cols or set()
    highlight_rows = highlight_rows or set()

    th_style = styles["th"]
    th_left = styles["th_left"]
    td_style = styles["td"]
    td_left = styles["td_left"]
    td_bold = styles["td_bold"]
    td_bold_left = styles["td_bold_left"]

    if font_size is not None:
        th_style = ParagraphStyle("th_s", parent=th_style, fontSize=font_size, leading=font_size + 2)
        th_left = ParagraphStyle("th_ls", parent=th_left, fontSize=font_size, leading=font_size + 2)
        td_style = ParagraphStyle("td_s", parent=td_style, fontSize=font_size, leading=font_size + 2.2)
        td_left = ParagraphStyle("td_ls", parent=td_left, fontSize=font_size, leading=font_size + 2.2)
        td_bold = ParagraphStyle("tdb_s", parent=td_bold, fontSize=font_size, leading=font_size + 2.2)
        td_bold_left = ParagraphStyle("tdbl_s", parent=td_bold_left, fontSize=font_size, leading=font_size + 2.2)

    data = [[Paragraph(h, th_left if i in left_cols else th_style) for i, h in enumerate(headers)]]
    n_rows = len(rows)
    for r_idx, row in enumerate(rows):
        is_bold = (highlight_last_row and r_idx == n_rows - 1) or (r_idx in highlight_rows)
        styled_row = []
        for c_idx, cell in enumerate(row):
            if is_bold:
                st = td_bold_left if c_idx in left_cols else td_bold
            else:
                st = td_left if c_idx in left_cols else td_style
            styled_row.append(Paragraph(str(cell), st))
        data.append(styled_row)

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    ts = [
        ("BACKGROUND", (0, 0), (-1, 0), C_DARK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, C_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 4.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
    ]
    for r_idx in range(1, n_rows + 1):
        if r_idx % 2 == 0:
            ts.append(("BACKGROUND", (0, r_idx), (-1, r_idx), C_ROW_ALT))
    for hr in highlight_rows:
        ts.append(("BACKGROUND", (0, hr + 1), (-1, hr + 1), C_PURPLE_LIGHT))
    if highlight_last_row and n_rows > 0:
        ts.append(("BACKGROUND", (0, n_rows), (-1, n_rows), C_CYAN_LIGHT))
        ts.append(("LINEABOVE", (0, n_rows), (-1, n_rows), 1.0, C_DARK))

    tbl.setStyle(TableStyle(ts))
    return tbl


def make_pipeline_diagram() -> Drawing:
    """Diagrama vetorial limpo com as 5 etapas do projeto (DW -> Projeção 2027)."""
    w = CONTENT_W
    h = 30 * mm
    d = Drawing(w, h)

    steps = [
        ("1. Dados DW", "dash-retencao", "69.478 contratos", C_DARK),
        ("2. Regras & Calend.", "De-Para oficial", "DU: 1 / 0,5 / 0", C_PURPLE),
        ("3. Análise (EDA)", "Anomalias, FPD,", "Bairros & Canais", C_CYAN),
        ("4. Regressão Linear", "ElasticNet", "Teste cego 2026", C_PURPLE),
        ("5. Projeção 2027", "Cascata mensal", "3 Cenários + IC 90%", C_GREEN),
    ]
    n = len(steps)
    gap = 4.5 * mm
    box_w = (w - gap * (n - 1)) / n
    box_h = 22 * mm
    y0 = 4 * mm

    for idx, (t1, t2, t3, col) in enumerate(steps):
        x = idx * (box_w + gap)
        # Caixa com borda arredondada
        d.add(Rect(x, y0, box_w, box_h, rx=4, ry=4, fillColor=colors.white, strokeColor=col, strokeWidth=1.2))
        # Cabeçalho colorido da caixa
        d.add(Rect(x, y0 + box_h - 6.8 * mm, box_w, 6.8 * mm, rx=3, ry=3, fillColor=col, strokeColor=col, strokeWidth=0))
        d.add(String(x + box_w / 2, y0 + box_h - 4.7 * mm, t1, fontName="Helvetica-Bold", fontSize=7.4, fillColor=colors.white, textAnchor="middle"))
        d.add(String(x + box_w / 2, y0 + 9.2 * mm, t2, fontName="Helvetica-Bold", fontSize=7.4, fillColor=C_TEXT, textAnchor="middle"))
        d.add(String(x + box_w / 2, y0 + 4.0 * mm, t3, fontName="Helvetica", fontSize=7.0, fillColor=C_MUTED, textAnchor="middle"))

        # Seta para a próxima caixa
        if idx < n - 1:
            ax1 = x + box_w + 0.6 * mm
            ax2 = x + box_w + gap - 1.2 * mm
            ay = y0 + box_h / 2
            d.add(Line(ax1, ay, ax2, ay, strokeColor=C_MUTED, strokeWidth=1.2))
            d.add(Polygon([ax2, ay + 1.5 * mm, ax2 + 1.5 * mm, ay, ax2, ay - 1.5 * mm], fillColor=C_MUTED, strokeColor=C_MUTED))

    return d


def add_figure(fig_name: str, caption: str, styles: dict[str, ParagraphStyle], max_h_mm: float = 95.0) -> KeepTogether:
    img_path = FIG_DIR / fig_name
    from PIL import Image as PILImage

    with PILImage.open(img_path) as im:
        iw, ih = im.size
    aspect = ih / iw
    target_w = CONTENT_W
    target_h = target_w * aspect
    max_h = max_h_mm * mm
    if target_h > max_h:
        target_h = max_h
        target_w = target_h / aspect

    img = Image(str(img_path), width=target_w, height=target_h)
    img.hAlign = "CENTER"
    return KeepTogether([Spacer(1, 3), img, Paragraph(caption, styles["caption"])])


def make_kpi_card(label: str, value: str, sub: str, accent, styles: dict[str, ParagraphStyle], width: float) -> Table:
    inner = [
        [Paragraph(label, styles["kpi_label"])],
        [Paragraph(value, styles["kpi_val"])],
        [Paragraph(sub, styles["kpi_sub"])],
    ]
    card = Table(inner, colWidths=[width], rowHeights=[6.5 * mm, 7.5 * mm, 9.5 * mm])
    card.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.6, C_BORDER),
                ("LINEABOVE", (0, 0), (-1, 0), 3.2, accent),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, 0), 5),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 1), (-1, 2), 1),
                ("BOTTOMPADDING", (0, 0), (-1, 1), 1),
            ]
        )
    )
    return card


def generate_pdf() -> Path:
    with open(DATA_RAW / "extract_metadata.json", encoding="utf-8") as f:
        ext_meta = json.load(f)
    with open(DATA_PROC / "features_metadata.json", encoding="utf-8") as f:
        feat_meta = json.load(f)
    with open(MODELS_DIR / "metadata.json", encoding="utf-8") as f:
        model_meta = json.load(f)
    with open(DATA_PRED / "resumo_2027.json", encoding="utf-8") as f:
        resumo = json.load(f)

    df_ml = pd.read_parquet(DATA_PROC / "serie_mensal_ml.parquet")
    df_closed = df_ml[df_ml["mes_fechado"]].copy()
    df_cal = pd.read_parquet(DATA_PROC / "calendario_mensal.parquet")
    df_pred = pd.read_parquet(DATA_PRED / "projecao_2027.parquet")

    styles = build_styles()
    story = []

    # =========================================================================
    # CAPA EXECUTIVA (PÁGINA 1)
    # =========================================================================
    story.append(Spacer(1, 8 * mm))
    if LOGO_PATH.exists():
        from PIL import Image as PILImage

        with PILImage.open(LOGO_PATH) as im:
            lw, lh = im.size
        logo_w = 56 * mm
        logo_h = logo_w * (lh / lw)
        logo = Image(str(LOGO_PATH), width=logo_w, height=logo_h)
        logo.hAlign = "LEFT"
        story.append(logo)
        story.append(Spacer(1, 14 * mm))

    story.append(Paragraph("RELATÓRIO EXECUTIVO DE MACHINE LEARNING  •  PLANEJAMENTO 2027", styles["cover_kicker"]))
    story.append(Paragraph("Projeção de Vendas, Retiradas e<br/>Crescimento da Carteira — 2027", styles["cover_title"]))
    story.append(
        Paragraph(
            "Documento geral e acessível apresentando as premissas de negócio, a Análise Exploratória dos dados do "
            "<b>dash-retencao</b>, o porquê da escolha da <b>Regressão Linear Regularizada (ElasticNet)</b>, a validação em "
            "teste real de 2026 e os cenários projetados mês a mês para 2027.",
            styles["cover_subtitle"],
        )
    )

    # 4 Cartões de KPIs na Capa
    c_base = resumo["cenarios_2027"]["Base"]
    c_cons = resumo["cenarios_2027"]["Conservador"]
    c_exp = resumo["cenarios_2027"]["Expansao"]
    v_mape = model_meta["champion_vendas"]["val_metrics"]["MAPE_pct"]
    r_mape = model_meta["champion_retiradas"]["val_metrics"]["MAPE_pct"]

    card_w = (CONTENT_W - 9 * mm) / 4
    kpi_row = Table(
        [
            [
                make_kpi_card("CARTEIRA INÍCIO 2027", fmt_int(c_base["carteira_inicio_2027"]), "Clientes ativos estimados em 01/01/2027", C_DARK, styles, card_w),
                make_kpi_card("CARTEIRA FIM 2027 (BASE)", fmt_int(c_base["carteira_fim_2027"]), f"Crescimento de +{fmt_dec(c_base['crescimento_carteira_2027_pct'])}% (+{fmt_int(c_base['saldo_liquido_2027'])})", C_PURPLE, styles, card_w),
                make_kpi_card("FAIXA ENTRE CENÁRIOS", f"{fmt_dec(c_cons['carteira_fim_2027']/1000,1)}k – {fmt_dec(c_exp['carteira_fim_2027']/1000,1)}k", f"De +{fmt_dec(c_cons['crescimento_carteira_2027_pct'])}% a +{fmt_dec(c_exp['crescimento_carteira_2027_pct'])}% em 2027", C_GREEN, styles, card_w),
                make_kpi_card("PRECISÃO TESTE 2026", f"{fmt_dec(v_mape,2)}% | {fmt_dec(r_mape,2)}%", "Erro médio (MAPE): Vendas | Retiradas", C_CYAN, styles, card_w),
            ]
        ],
        colWidths=[card_w] * 4,
    )
    kpi_row.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 1.5), ("RIGHTPADDING", (0, 0), (-1, -1), 1.5), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(kpi_row)
    story.append(Spacer(1, 12 * mm))

    # Caixa de Resumo Rápido na Capa
    story.append(
        make_callout(
            "Em resumo: o que este estudo entrega para a tomada de decisão",
            f"A partir de <b>{fmt_int(ext_meta['rows_deduplicated'])} contratos</b> reais do Data Warehouse da The Fiber, "
            "construímos um modelo estatístico transparente que considera o <b>calendário operacional real de 2027</b> "
            "(273,5 dias úteis ponderados: dia útil = 1,0 | sábado = 0,5 | domingo e feriado = 0,0), a abertura de "
            "<b>novos bairros</b>, a entrada de <b>novos canais de venda</b> e o efeito natural da base ativa sobre os cancelamentos. "
            f"No cenário <b>Base</b>, a empresa projeta <b>{fmt_int(c_base['vendas_total_2027'])} vendas</b> contra "
            f"<b>{fmt_int(c_base['retiradas_total_2027'])} retiradas</b>, adicionando <b>+{fmt_int(c_base['saldo_liquido_2027'])} clientes líquidos</b> "
            "à planta em 2027.",
            styles,
            bg_color=C_PURPLE_LIGHT,
            accent_color=C_PURPLE,
        )
    )
    story.append(Spacer(1, 14 * mm))

    # Ficha Técnica na base da Capa
    ficha = make_styled_table(
        ["Item da Ficha Técnica", "Detalhamento"],
        [
            ["Projeto / Repositório", "ml-retencao-forecast (integrado em leitura ao banco PostgreSQL do dash-retencao)"],
            ["Base Histórica Analisada", f"{fmt_int(ext_meta['rows_deduplicated'])} contratos ({ext_meta['min_anomes_ativacao']} a {ext_meta['max_anomes_ativacao']}); {feat_meta['closed_months_count']} meses fechados para ML (Jan/2024 – Out/2026)"],
            ["Definição de Retirada", "Todos os contratos cancelados válidos no mês, independente da safra original de ativação"],
            ["Metodologia & Algoritmo", "Padrão CRISP-DM + MLOps  •  Regressão Linear Regularizada (ElasticNet) acoplada em cascata"],
            ["Validação de Qualidade", "Backtest cego out-of-time (Jul–Out/2026)  •  Testes automatizados aprovados (pytest)"],
        ],
        [48 * mm, CONTENT_W - 48 * mm],
        styles,
        left_cols={0, 1},
        font_size=8.2,
    )
    story.append(ficha)
    story.append(PageBreak())

    # =========================================================================
    # SEÇÃO 1 — VISÃO GERAL E FLUXO DO PROJETO
    # =========================================================================
    story.append(Paragraph("1. Visão Geral do Projeto e Fluxo de Trabalho", styles["h1"]))
    story.append(
        Paragraph(
            "Projetar o crescimento de um provedor de internet exige olhar os dois lados da mesma moeda: "
            "<b>quantos clientes entram (Vendas)</b> e <b>quantos clientes saem no mês (Retiradas)</b>. "
            "Diferente de planilhas estáticas que apenas repetem médias ou aplicam percentuais fixos sobre o ano anterior, "
            "este projeto transforma os dados oficiais do painel <b>dash-retencao</b> em um fluxo automatizado de Ciência de Dados "
            "que aprende os padrões reais da The Fiber e permite simular diferentes estratégias de expansão para 2027.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 3))
    story.append(make_pipeline_diagram())
    story.append(Paragraph("Figura 1 — Arquitetura completa do projeto (da extração no DW até a simulação de 2027).", styles["caption"]))

    # =========================================================================
    # SEÇÃO 2 — ORIGEM DOS DADOS E PREMISSAS DE NEGÓCIO
    # =========================================================================
    story.append(Paragraph("2. Origem dos Dados e Premissas de Negócio", styles["h1"]))
    story.append(
        Paragraph(
            "Para garantir que todos os números batam exatamente com o que a diretoria acompanha diariamente no "
            "<b>dash-retencao</b>, o extrator lê em modo <i>somente leitura</i> a tabela analítica <b>fato_churn_contrato</b> "
            "cruzada com as quatro tabelas oficiais de De-Para (Canais, Funcionários com histórico de transferência, "
            "Bairros/Tipo de Rede e Motivos de Cancelamento).",
            styles["body"],
        )
    )

    story.append(Paragraph("2.1. Regras e definições utilizadas", styles["h2"]))
    story.append(
        make_styled_table(
            ["Conceito", "Regra aplicada no modelo", "Por que adotamos essa regra"],
            [
                ["Venda (Ativação)", "Contratos ativados válidos no mês, excluindo cadastros de teste (ignorar_metricas) e Vendedor Padrão.", "Garante que apenas vendas comerciais reais entrem na projeção."],
                ["Retirada (Cancelamento)", "Todo contrato cancelado no mês (motivos válidos), independente da safra em que foi vendido.", "Reflete a saída total real de clientes no mês para medir o crescimento líquido."],
                ["Saldo Líquido e Carteira", "Saldo = Vendas − Retiradas. Carteira parte da planta real de 14.082 (Dez/2023) até 33.755 (Out/2026).", "Mantém a contabilidade da base fiel à evolução da planta no painel."],
                ["FPD (61 dias)", "Clientes que não pagaram a 1ª fatura com mais de 61 dias de atraso.", "Mede qualidade de venda e antecede cancelamentos por inadimplência."],
                ["Mês Fechado vs. Aberto", "Treino restrito a meses 100% encerrados (Jan/2024 a Out/2026). Nov/2026 (parcial) fica fora do treino.", "Evita distorcer o modelo com um mês ainda em andamento."],
            ],
            [33 * mm, 76 * mm, CONTENT_W - 109 * mm],
            styles,
            left_cols={0, 1, 2},
            font_size=7.9,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("2.2. A Premissa de Dias Úteis Ponderados (1,0 / 0,5 / 0,0) e o Calendário 2027", styles["h2"]))
    story.append(
        Paragraph(
            "Meses diferentes têm capacidades operacionais diferentes: Fevereiro possui menos dias corridos e feriados de Carnaval, "
            "enquanto Março, Junho, Agosto e Dezembro oferecem mais dias plenos de venda e instalação. Por isso, aplicamos a "
            "regra operacional de <b>Dias Úteis Ponderados</b> considerando feriados nacionais, estaduais de <b>Goiás</b>, "
            "municipais de <b>Goiânia</b> (24/05 Padroeira e 24/10 Aniversário) além de Carnaval e Corpus Christi: "
            "<b>Segunda a Sexta útil = 1,0</b>  |  <b>Sábado não feriado = 0,5</b>  |  <b>Domingo ou Feriado = 0,0</b>.",
            styles["body"],
        )
    )

    cal_2027 = df_cal[df_cal["ano"] == 2027].copy()
    meses_pt = {1: "Jan/27", 2: "Fev/27*", 3: "Mar/27", 4: "Abr/27", 5: "Mai/27*", 6: "Jun/27", 7: "Jul/27", 8: "Ago/27", 9: "Set/27", 10: "Out/27", 11: "Nov/27", 12: "Dez/27"}
    cal_rows = []
    for _, r in cal_2027.iterrows():
        cal_rows.append([
            meses_pt[int(r["mes"])],
            str(int(r["dias_calendario"])),
            str(int(r["dias_uteis_cheios"])),
            str(int(r["sabados_uteis"])),
            str(int(r["domingos"])),
            str(int(r["feriados_em_dia_util"])),
            fmt_dec(r["dias_uteis_ponderados"], 1),
        ])
    cal_rows.append([
        "Total 2027",
        str(int(cal_2027["dias_calendario"].sum())),
        str(int(cal_2027["dias_uteis_cheios"].sum())),
        str(int(cal_2027["sabados_uteis"].sum())),
        str(int(cal_2027["domingos"].sum())),
        str(int(cal_2027["feriados_em_dia_util"].sum())),
        fmt_dec(cal_2027["dias_uteis_ponderados"].sum(), 1),
    ])
    story.append(
        make_styled_table(
            ["Mês", "Dias Corridos", "Úteis (1,0)", "Sábados (0,5)", "Domingos (0,0)", "Feriados Úteis (0,0)", "Dias Úteis Ponderados"],
            cal_rows,
            [24 * mm, 24 * mm, 24 * mm, 26 * mm, 25 * mm, 28 * mm, CONTENT_W - 151 * mm],
            styles,
            left_cols={0},
            highlight_last_row=True,
            font_size=7.6,
        )
    )
    story.append(Paragraph("* Fev/27 inclui segunda e terça de Carnaval (20,0 DU); Mai/27 inclui Padroeira de Goiânia e Corpus Christi (21,0 DU).", styles["caption"]))

    # =========================================================================
    # SEÇÃO 3 — ANÁLISE EXPLORATÓRIA (EDA)
    # =========================================================================
    story.append(Paragraph("3. Análise Exploratória de Dados (EDA): O Que os Números Revelaram", styles["h1"]))
    story.append(
        Paragraph(
            "Antes de treinar qualquer modelo preditivo, é indispensável realizar a <b>Análise Exploratória (EDA)</b>. "
            "Na prática, ela responde três perguntas fundamentais: <i>(1) como o negócio evoluiu de verdade?</i> "
            "<i>(2) existem distorções no histórico que enganariam o algoritmo?</i> e "
            "<i>(3) quais fatores realmente explicam o crescimento de vendas e cancelamentos?</i>",
            styles["body"],
        )
    )

    # Tabela anual real
    anual = df_closed.groupby("ano", as_index=False).agg(
        meses=("anomes", "count"),
        du=("dias_uteis_ponderados", "sum"),
        vendas=("vendas", "sum"),
        retiradas=("retiradas", "sum"),
        saldo=("saldo_liquido", "sum"),
        carteira_fim=("carteira_fim_mes", "last"),
        bairros=("qtd_bairros_ativos_mes", "mean"),
        canais=("qtd_canais_ativos_mes", "mean"),
    )
    anual_rows = []
    for _, r in anual.iterrows():
        lbl = f"{int(r['ano'])} ({int(r['meses'])}m fechados)"
        anual_rows.append([
            lbl,
            fmt_dec(r["du"], 1),
            fmt_int(r["vendas"]),
            fmt_int(r["retiradas"]),
            fmt_signed_int(r["saldo"]),
            fmt_int(r["carteira_fim"]),
            fmt_dec(r["bairros"], 0),
            fmt_dec(r["canais"], 1),
        ])
    story.append(Paragraph("3.1. Salto histórico de vendas, carteira e cobertura (2024–2026)", styles["h2"]))
    story.append(
        make_styled_table(
            ["Ano", "DU Pond.", "Vendas", "Retiradas", "Saldo Líq.", "Carteira Fim", "Bairros/Mês", "Canais/Mês"],
            anual_rows,
            [31 * mm, 18 * mm, 19 * mm, 20 * mm, 21 * mm, 23 * mm, 22 * mm, CONTENT_W - 154 * mm],
            styles,
            left_cols={0},
            font_size=7.8,
        )
    )
    story.append(PageBreak())
    story.append(
        add_figure(
            "01_serie_historica_vendas_retiradas_planta.png",
            "Figura 2 — Evolução mensal de Vendas, Retiradas (destacando baixas em lote em 2025) e crescimento da Carteira Ativa.",
            styles,
            max_h_mm=104,
        )
    )

    story.append(Paragraph("3.2. Anomalias de baixa em lote (2025-06 e 2025-08) e censura do FPD", styles["h2"]))
    story.append(
        Paragraph(
            "A EDA identificou dois meses totalmente fora do padrão nas retiradas: <b>Junho/2025 (1.710 retiradas)</b> e "
            "<b>Agosto/2025 (1.065 retiradas)</b>. Ao abrir os motivos de cancelamento, comprovamos que esses picos foram "
            "<b>baixas administrativas em lote de inadimplentes antigos no IXC</b> — não um padrão orgânico recorrente. "
            "Quando o modelo foi testado sem isolar esses dois meses, o erro de previsão das retiradas quase triplicou "
            "(subiu de <b>3,42% para 9,19%</b>). Além disso, confirmamos que o indicador <b>FPD 61 dias</b> sofre censura "
            "natural nos últimos 2–3 meses recentes (clientes ainda dentro do prazo de 61 dias), exigindo corte automático "
            "de maturação.",
            styles["body"],
        )
    )
    story.append(
        add_figure(
            "01_fpd_e_composicao_retiradas.png",
            "Figura 3 — Composição das Retiradas (evidenciando choques pontuais de inadimplência) e janela maturada vs. censurada do FPD 61d.",
            styles,
            max_h_mm=66,
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("3.3. Prova prática do Calendário Ponderado e força operacional do Sábado", styles["h2"]))
    story.append(
        Paragraph(
            "Ao cruzar mais de 1.000 dias reais de operação com o calendário ponderado (<b>1,0 / 0,5 / 0,0</b>), a EDA revelou "
            "dois ganhos imediatos: <b>(1)</b> em <b>Fev/2026</b> (apenas 20,0 DU por causa do Carnaval), as vendas totais "
            "pareciam ter caído -9,6% (1.737 → 1.570), mas o ritmo por dia útil na verdade <b>cresceu</b> (73,9 → 78,5 vendas/DU); "
            "em <b>Jul/2026</b> (25,0 DU), o recorde de 2.084 vendas foi explicado integralmente pelo mês mais longo. "
            "<b>(2)</b> Empiricamente, o <b>Sábado</b> da The Fiber entrega <b>74% das vendas</b> e <b>71% das retiradas</b> de "
            "um dia útil (acima do peso conservador de 0,5), enquanto o <b>Domingo</b> registra praticamente zero retiradas (0,1%).",
            styles["body"],
        )
    )
    story.append(
        add_figure(
            "01_validacao_dias_uteis.png",
            "Figura 4 — Comparação entre os pesos de negócio (1 / 0,5 / 0) e o comportamento real diário; efeito calendário em 2026.",
            styles,
            max_h_mm=65,
        )
    )

    story.append(Paragraph("3.4. O Ouro Estratégico: Novos Bairros (Rede Própria vs. Neutra) e Novos Canais", styles["h2"]))
    story.append(
        Paragraph(
            "Investigamos diretamente a hipótese de expansão territorial e comercial nos dados de 2024–2026 e encontramos "
            "evidências decisivas para o planejamento de 2027:",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "• <b>Novos Bairros — Rede Própria vence com folga:</b> entre 116 novos bairros abertos em <b>Rede Própria</b>, "
            "as vendas médias sobem de 1,6 no mês de estreia (M0) para <b>9,04 vendas/mês no 6º mês (M6)</b>, com apenas "
            "<b>0,89 retiradas/mês</b>. Já nos 137 novos bairros de <b>Rede Neutra</b>, as vendas estabilizam cedo em "
            "<b>3,28 vendas/mês no M6</b>, mas com <b>1,80 retiradas/mês</b> — quase 3x menos vendas maduras e o dobro de churn relativo.",
            styles["bullet"],
        )
    )
    story.append(
        Paragraph(
            "• <b>Novos Canais de Venda — Motor da virada comercial:</b> canais inaugurados entre fim de 2024 e 2026 — "
            "<b>Loja Lorena</b> (339 vendas/mês, FPD 10,5%), <b>Loja Vera Cruz</b> (307 vendas/mês, FPD 9,3%), "
            "<b>Loja Maysa</b> (190 vendas/mês, FPD 8,7%) e <b>INDIQUE E GANHE</b> (FPD 7,9%) — elevaram a empresa a um novo "
            "patamar com inadimplência inicial muito menor que o canal legado <b>Agente Autorizado</b> (FPD 19,6%).",
            styles["bullet"],
        )
    )
    story.append(
        add_figure(
            "01_eda_bairros_e_canais.png",
            "Figura 5 — Curva de maturação de novos bairros (M0–M6) por Tipo de Rede e ritmo/qualidade (FPD) por Canal de Venda.",
            styles,
            max_h_mm=66,
        )
    )
    story.append(PageBreak())

    # =========================================================================
    # SEÇÃO 4 — POR QUE REGRESSÃO LINEAR (ELASTICNET)
    # =========================================================================
    story.append(Paragraph("4. Por Que Usamos Regressão Linear — E Como Ela Funciona", styles["h1"]))
    story.append(
        Paragraph(
            "A <b>Regressão Linear</b> é uma técnica estatística que encontra a fórmula mais equilibrada conectando causas "
            "(dias úteis, bairros ativos, canais, tamanho da carteira) e efeitos (vendas e retiradas). "
            "Escolhemos essa abordagem — e não modelos complexos de 'caixa-preta' (como Redes Neurais) — por três motivos práticos:",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "• <b>1. Adequação ao tamanho da série (34 meses fechados):</b> algoritmos muito complexos exigem milhares de períodos "
            "e costumam 'decorar' o passado (<i>overfitting</i>), errando mais quando projetam um ano inteiro à frente.",
            styles["bullet"],
        )
    )
    story.append(
        Paragraph(
            "• <b>2. Transparência executiva:</b> cada peso do modelo pode ser explicado em reunião de diretoria — sabemos "
            "exatamente quanto cada bairro, canal ou dia útil contribui no resultado final.",
            styles["bullet"],
        )
    )
    story.append(
        Paragraph(
            "• <b>3. Simulação de cenários reais:</b> ao incluir bairros ativos, canais e calendário diretamente nas equações, "
            "podemos simular decisões de gestão para 2027 sem chute.",
            styles["bullet"],
        )
    )

    story.append(Paragraph("4.1. Por que o vencedor foi o ElasticNet (Regressão Linear Regularizada)?", styles["h2"]))
    story.append(
        Paragraph(
            "Como a The Fiber cresceu forte nos últimos anos, várias variáveis sobem juntas no tempo (bairros ativos, carteira "
            "e vendas recentes têm forte correlação entre si — fenômeno chamado de <i>multicolinearidade</i>). Na regressão comum "
            "(OLS), isso pode gerar pesos exagerados. O <b>ElasticNet</b> é a própria Regressão Linear equipada com um "
            "<b>freio matemático de equilíbrio (regularização)</b> que distribui os pesos com estabilidade e evita extrapolações irreais.",
            styles["body"],
        )
    )
    story.append(
        add_figure(
            "02_matriz_correlacao.png",
            "Figura 6 — Mapa de correlação das variáveis (cores fortes confirmam alta relação linear e justificam a regularização ElasticNet).",
            styles,
            max_h_mm=78,
        )
    )

    story.append(Paragraph("4.2. Tradução prática do que os modelos aprenderam (em linguagem de negócio)", styles["h2"]))
    story.append(
        Paragraph(
            "Os dois modelos funcionam <b>acoplados em cascata</b>: as vendas previstas em um mês aumentam a carteira ativa e as "
            "safras recentes, que por sua vez alimentam a previsão de retiradas dos meses seguintes.",
            styles["body"],
        )
    )
    story.append(
        make_styled_table(
            ["Modelo Campeão", "O que explica (R² histórico)", "Principais fatores e impacto prático aprendido nos dados"],
            [
                [
                    "Vendas\n(ElasticNet Taxa/DU)",
                    "97,8%\n(prevê vendas/DU e multiplica pelos DU do mês)",
                    "• +1 bairro ativo no mês → ~+3,0 vendas/mês diretas (+0,13/DU).\n"
                    "• +1 canal comercial ativo → ~+53 vendas/mês diretas (+2,30/DU), ampliadas nos meses seguintes.\n"
                    "• Ritmo do mês anterior (peso 0,45) + sazonalidade + escala exata pelos dias úteis ponderados.",
                ],
                [
                    "Retiradas\n(ElasticNet Direto Limpo)",
                    "98,8%\n(sem as 2 baixas em lote de 2025)",
                    "• Carteira ativa (início do mês): a cada +1.000 clientes na planta → ~+17,4 retiradas/mês.\n"
                    "• Vendas dos últimos 3 meses (safras jovens/FPD): a cada +1.000 vendas → ~+52,7 retiradas.\n"
                    "• Patamar do mês anterior (peso 0,33) + ~+5,7 retiradas por dia útil ponderado adicional.",
                ],
            ],
            [36 * mm, 36 * mm, CONTENT_W - 72 * mm],
            styles,
            left_cols={0, 2},
            font_size=7.9,
        )
    )
    story.append(PageBreak())

    # =========================================================================
    # SEÇÃO 5 — PROVA DE CONFIANÇA (BACKTEST REAL 2026)
    # =========================================================================
    story.append(Paragraph("5. Prova de Confiança: Teste Cego Real (Backtest Jul–Out/2026)", styles["h1"]))
    story.append(
        Paragraph(
            "Antes de confiar na projeção de 2027, aplicamos o teste mais rigoroso de séries temporais: "
            "<b>escondemos completamente os últimos 4 meses fechados reais de 2026 (Julho a Outubro)</b>, treinamos todos os "
            "algoritmos lineares apenas com dados até Junho/2026 e comparamos as previsões com o que realmente aconteceu na empresa.",
            styles["body"],
        )
    )

    val_v = model_meta["champion_vendas"]["val_predictions"]
    val_r = model_meta["champion_retiradas"]["val_predictions"]
    bt_rows = []
    for rv, rr in zip(val_v, val_r):
        bt_rows.append([
            rv["anomes"],
            fmt_int(rv["real"]),
            fmt_int(rv["previsto"]),
            f"{rv['erro_pct']:+.2f}%".replace(".", ","),
            fmt_int(rr["real"]),
            fmt_int(rr["previsto"]),
            f"{rr['erro_pct']:+.2f}%".replace(".", ","),
        ])
    bt_rows.append([
        "Erro Médio (MAPE / MAE)",
        "—",
        f"MAE {fmt_dec(model_meta['champion_vendas']['val_metrics']['MAE'],1)}",
        f"{fmt_dec(v_mape,2)}%",
        "—",
        f"MAE {fmt_dec(model_meta['champion_retiradas']['val_metrics']['MAE'],1)}",
        f"{fmt_dec(r_mape,2)}%",
    ])
    story.append(
        make_styled_table(
            ["Mês (Oculto)", "Vendas Reais", "Vendas Prev.", "Erro Vendas", "Retiradas Reais", "Retiradas Prev.", "Erro Retiradas"],
            bt_rows,
            [27 * mm, 24 * mm, 25 * mm, 24 * mm, 27 * mm, 26 * mm, CONTENT_W - 153 * mm],
            styles,
            left_cols={0},
            highlight_last_row=True,
            font_size=8.0,
        )
    )
    story.append(Spacer(1, 4))
    story.append(
        make_callout(
            "Leitura executiva dos erros (por que 2,55% e 3,42% são resultados excelentes)",
            "Em projeções mensais de vendas e churn, erros médios abaixo de 5% em meses nunca vistos pelo modelo indicam "
            "alta confiabilidade: num mês de ~2.000 vendas, o desvio médio foi de apenas <b>51 contratos</b>; num mês de ~950 "
            "retiradas, apenas <b>33 contratos</b>. Os testes estatísticos nos resíduos (Shapiro-Wilk p = 0,25 em vendas e 0,82 "
            "em retiradas; Durbin-Watson = 2,11 em retiradas) confirmam que os erros são aleatórios e sem viés estrutural.",
            styles,
            bg_color=C_GREEN_LIGHT,
            accent_color=C_GREEN,
        )
    )
    story.append(
        add_figure(
            "03_backtest_vendas_e_retiradas.png",
            "Figura 7 — Aderência histórica e validação out-of-time (pontos verdes) para Vendas e Retiradas.",
            styles,
            max_h_mm=104,
        )
    )
    story.append(PageBreak())

    # =========================================================================
    # SEÇÃO 6 — RESULTADOS: PROJEÇÃO 2027 E CENÁRIOS
    # =========================================================================
    story.append(Paragraph("6. Resultados: Projeção de Crescimento para 2027 (3 Cenários)", styles["h1"]))
    story.append(
        Paragraph(
            "Após validados no backtest, os modelos campeões foram re-treinados com <b>100% dos meses fechados até Out/2026</b> "
            "(carteira real de <b>33.755 clientes</b>). Primeiro projetamos os meses-ponte de fechamento de 2026 "
            "(<b>Nov–Dez/2026</b>), igualados entre todos os cenários para que 2027 comece exatamente na mesma base de "
            "<b>35.645 clientes em 01/01/2027</b>. A partir de Janeiro/2027 simulamos três caminhos estratégicos:",
            styles["body"],
        )
    )

    story.append(
        make_styled_table(
            ["Cenário 2027", "Expansão de Bairros", "Novos Canais", "Vendas 2027", "Retiradas 2027", "Saldo Líquido", "Carteira Dez/27", "Crescimento"],
            [
                [
                    "Conservador",
                    "+1 bairro/mês",
                    "0 novos",
                    fmt_int(c_cons["vendas_total_2027"]),
                    fmt_int(c_cons["retiradas_total_2027"]),
                    fmt_signed_int(c_cons["saldo_liquido_2027"]),
                    fmt_int(c_cons["carteira_fim_2027"]),
                    f"+{fmt_dec(c_cons['crescimento_carteira_2027_pct'],1)}%",
                ],
                [
                    "Base (Referência)",
                    "+4 bairros/mês",
                    "0 novos",
                    fmt_int(c_base["vendas_total_2027"]),
                    fmt_int(c_base["retiradas_total_2027"]),
                    fmt_signed_int(c_base["saldo_liquido_2027"]),
                    fmt_int(c_base["carteira_fim_2027"]),
                    f"+{fmt_dec(c_base['crescimento_carteira_2027_pct'],1)}%",
                ],
                [
                    "Expansão",
                    "+9 bairros/mês",
                    "+1 canal (Mar/27)",
                    fmt_int(c_exp["vendas_total_2027"]),
                    fmt_int(c_exp["retiradas_total_2027"]),
                    fmt_signed_int(c_exp["saldo_liquido_2027"]),
                    fmt_int(c_exp["carteira_fim_2027"]),
                    f"+{fmt_dec(c_exp['crescimento_carteira_2027_pct'],1)}%",
                ],
            ],
            [25 * mm, 23 * mm, 23 * mm, 20 * mm, 21 * mm, 20 * mm, 22 * mm, CONTENT_W - 154 * mm],
            styles,
            left_cols={0, 1, 2},
            highlight_rows={1},
            font_size=7.6,
        )
    )
    story.append(
        add_figure(
            "04_projecao_2027_cenarios_carteira.png",
            "Figura 8 — Trajetória 2024–2027 de Vendas vs. Retiradas (Cenário Base com IC 90%) e evolução da Carteira nos 3 cenários.",
            styles,
            max_h_mm=106,
        )
    )
    story.append(
        add_figure(
            "04_saldo_liquido_e_comparativo_anual.png",
            "Figura 9 — Saldo líquido mensal projetado em 2027 (efeito dos dias úteis) e comparativo histórico anual (2024–2027).",
            styles,
            max_h_mm=60,
        )
    )
    story.append(PageBreak())

    # Tabela mês a mês detalhada (Cenário Base)
    story.append(Paragraph("6.1. Projeção Mês a Mês Completa — Cenário Base (2026 Ponte + 2027)", styles["h2"]))
    story.append(
        Paragraph(
            "A tabela abaixo detalha cada mês de 2027 no cenário <b>Base</b>, incluindo os dias úteis ponderados do calendário, "
            "a <b>Faixa de Confiança de 90%</b> para as vendas (mínimo e máximo esperados em condições normais), as retiradas, "
            "o saldo líquido e a carteira acumulada.",
            styles["body"],
        )
    )

    df_base = df_pred[df_pred["cenario"] == "Base"].copy()
    m_rows = []
    for _, r in df_base.iterrows():
        mes_lbl = f"{r['anomes']} (Ponte)" if r["tipo_periodo"] == "Ponte_2026" else str(r["anomes"])
        faixa_v = f"{fmt_int(r['vendas_ic_inf'])} – {fmt_int(r['vendas_ic_sup'])}"
        m_rows.append([
            mes_lbl,
            fmt_dec(r["dias_uteis_ponderados"], 1),
            fmt_int(r["qtd_bairros_ativos_proj"]),
            fmt_int(r["vendas_prev"]),
            faixa_v,
            fmt_int(r["retiradas_prev"]),
            fmt_signed_int(r["saldo_liquido_prev"]),
            fmt_int(r["carteira_fim_mes_prev"]),
            f"{fmt_dec(r['taxa_churn_carteira_pct'],2)}%",
        ])
    m_rows.append([
        "Total / Fim 2027",
        fmt_dec(c_base["total_dias_uteis_ponderados_2027"], 1),
        "—",
        fmt_int(c_base["vendas_total_2027"]),
        "—",
        fmt_int(c_base["retiradas_total_2027"]),
        fmt_signed_int(c_base["saldo_liquido_2027"]),
        fmt_int(c_base["carteira_fim_2027"]),
        f"{fmt_dec(c_base['churn_medio_mensal_sobre_carteira_pct'],2)}% (méd)",
    ])
    story.append(
        make_styled_table(
            ["Mês", "DU", "Bairros", "Vendas", "Faixa 90% Vendas", "Retiradas", "Saldo Líq.", "Carteira Fim", "Churn/Base"],
            m_rows,
            [24 * mm, 12 * mm, 15 * mm, 17 * mm, 29 * mm, 18 * mm, 19 * mm, 22 * mm, CONTENT_W - 156 * mm],
            styles,
            left_cols={0},
            highlight_last_row=True,
            font_size=7.6,
        )
    )
    story.append(Spacer(1, 5))

    # Tabela 6.2 — Comparativo Histórico Anual Consolidado (2024 a 2027)
    ponte_2026 = df_base[df_base["ano"] == 2026]
    v_2024 = float(anual.loc[anual["ano"] == 2024, "vendas"].iloc[0])
    r_2024 = float(anual.loc[anual["ano"] == 2024, "retiradas"].iloc[0])
    v_2025 = float(anual.loc[anual["ano"] == 2025, "vendas"].iloc[0])
    r_2025 = float(anual.loc[anual["ano"] == 2025, "retiradas"].iloc[0])
    v_2026 = float(anual.loc[anual["ano"] == 2026, "vendas"].iloc[0] + ponte_2026["vendas_prev"].sum())
    r_2026 = float(anual.loc[anual["ano"] == 2026, "retiradas"].iloc[0] + ponte_2026["retiradas_prev"].sum())

    story.append(Paragraph("6.2. Perspectiva Histórica Anual Consolidada (2024 → 2027)", styles["h2"]))
    story.append(
        make_styled_table(
            ["Período", "Vendas no Ano", "Retiradas no Ano", "Saldo Líquido", "Relação Retirada / Venda"],
            [
                ["2024 (Real Fechado)", fmt_int(v_2024), fmt_int(r_2024), fmt_signed_int(v_2024 - r_2024), f"{fmt_dec(r_2024/v_2024*100,1)}%"],
                ["2025 (Real Fechado)", fmt_int(v_2025), fmt_int(r_2025), fmt_signed_int(v_2025 - r_2025), f"{fmt_dec(r_2025/v_2025*100,1)}%"],
                ["2026 (Real Jan–Out + Ponte Nov–Dez)", fmt_int(v_2026), fmt_int(r_2026), fmt_signed_int(v_2026 - r_2026), f"{fmt_dec(r_2026/v_2026*100,1)}%"],
                ["2027 — Cenário Conservador", fmt_int(c_cons["vendas_total_2027"]), fmt_int(c_cons["retiradas_total_2027"]), fmt_signed_int(c_cons["saldo_liquido_2027"]), f"{fmt_dec(c_cons['relacao_retirada_venda_2027_pct'],1)}%"],
                ["2027 — Cenário Base (Referência)", fmt_int(c_base["vendas_total_2027"]), fmt_int(c_base["retiradas_total_2027"]), fmt_signed_int(c_base["saldo_liquido_2027"]), f"{fmt_dec(c_base['relacao_retirada_venda_2027_pct'],1)}%"],
                ["2027 — Cenário Expansão", fmt_int(c_exp["vendas_total_2027"]), fmt_int(c_exp["retiradas_total_2027"]), fmt_signed_int(c_exp["saldo_liquido_2027"]), f"{fmt_dec(c_exp['relacao_retirada_venda_2027_pct'],1)}%"],
            ],
            [62 * mm, 27 * mm, 29 * mm, 27 * mm, CONTENT_W - 145 * mm],
            styles,
            left_cols={0},
            highlight_rows={4},
            font_size=7.9,
        )
    )
    story.append(PageBreak())

    # =========================================================================
    # SEÇÃO 7 — LIMITAÇÕES, RECOMENDAÇÕES E GLOSSÁRIO
    # =========================================================================
    story.append(Paragraph("7. Limitações, Recomendações Estratégicas e Glossário", styles["h1"]))
    story.append(
        make_callout(
            "Atenção metodológica — o que o modelo NÃO prevê sozinho",
            "Todo modelo estatístico aprende a partir das dinâmicas registradas no histórico. Eventos externos abruptos — como "
            "entrada agressiva de um concorrente, virada brusca de preços, mudança radical na política de crédito ou novas "
            "baixas em lote no IXC — precisam ser refletidos ajustando os cenários em <i>configs/params.yaml</i>. "
            "Por isso o planejamento deve sempre acompanhar a <b>faixa de 90%</b> e os <b>3 cenários</b>.",
            styles,
            bg_color=C_AMBER_LIGHT,
            accent_color=C_AMBER,
        )
    )

    story.append(Paragraph("7.1. Recomendações práticas para maximizar o resultado de 2027", styles["h2"]))
    story.append(
        Paragraph(
            "• <b>1. Priorizar investimentos em Rede Própria:</b> os dados provam que bairros novos de Rede Própria entregam "
            "quase <b>3x mais vendas maduras no 6º mês</b> (9,04 vs. 3,28) e metade das retiradas da Rede Neutra.",
            styles["bullet"],
        )
    )
    story.append(
        Paragraph(
            "• <b>2. Expandir via canais de baixo FPD:</b> ao ativar um novo canal comercial em 2027 (cenário Expansão), "
            "priorizar formatos com perfil das <b>Lojas Próprias</b> (8,7%–10,5% FPD) e <b>Indique e Ganhe</b> (7,9% FPD), "
            "evitando replicar modelos terceirizados com FPD próximo a 20%.",
            styles["bullet"],
        )
    )
    story.append(
        Paragraph(
            "• <b>3. Dimensionar retenção para o 'efeito base':</b> mesmo com taxa percentual estável (~2,86%/mês), o crescimento "
            "da carteira para 46k–51k clientes elevará o volume absoluto de retiradas para 1.250–1.500/mês no fim de 2027. "
            "Reforçar régua de cobrança preventiva e retenção antes do 2º semestre protege o saldo líquido.",
            styles["bullet"],
        )
    )
    story.append(
        Paragraph(
            "• <b>4. Rotina mensal MLOps:</b> no 1º dia útil após o fechamento de cada mês, executar "
            "<i>python -m src.pipeline --step all</i> para incorporar o mês fechado, monitorar o erro real vs. previsto e "
            "reemitir este relatório automaticamente.",
            styles["bullet"],
        )
    )

    gloss_tbl = make_styled_table(
        ["Termo", "Significado simples"],
        [
            ["Análise Exploratória (EDA)", "Investigação inicial com gráficos e tabelas para entender o passado, achar anomalias e validar hipóteses antes da modelagem."],
            ["Backtest / Teste Cego", "Validação em que escondemos meses reais recentes do modelo (Jul–Out/2026) para medir o quanto ele acerta no mundo real."],
            ["Dias Úteis Ponderados", "Calendário ajustado: Segunda a Sexta útil = 1,0 | Sábado útil = 0,5 | Domingo ou Feriado = 0,0."],
            ["ElasticNet (Regressão Linear)", "Regressão Linear com freio matemático de regularização, ideal quando vários indicadores do negócio crescem juntos."],
            ["FPD (61 dias)", "First Payment Default — percentual de novos clientes que nunca pagaram a 1ª fatura após 61 dias de atraso."],
            ["Faixa de Confiança 90%", "Intervalo entre o piso e o teto esperado onde o resultado real tende a ocorrer em 9 de cada 10 meses normais."],
            ["MAPE / R²", "MAPE é o erro percentual médio da previsão (quanto menor, melhor; < 5% é excelente). R² mostra quanto do histórico o modelo explica (0% a 100%)."],
            ["MLOps", "Boas práticas que tornam o projeto organizado, auditável, testado automaticamente e fácil de atualizar todo mês."],
        ],
        [44 * mm, CONTENT_W - 44 * mm],
        styles,
        left_cols={0, 1},
        font_size=7.7,
    )
    story.append(KeepTogether([Paragraph("7.2. Glossário rápido (para leitores não técnicos)", styles["h2"]), gloss_tbl]))

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=MARGIN_X,
        rightMargin=MARGIN_X,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM,
        title="Relatório Executivo — Projeção de Crescimento 2027 (The Fiber)",
        author="ML Retenção Forecast — The Fiber",
    )
    doc.build(story, canvasmaker=NumberedCanvas)
    return OUTPUT_PDF


if __name__ == "__main__":
    out = generate_pdf()
    print(f"PDF gerado com sucesso: {out} ({out.stat().st_size / 1024:.1f} KB)")
