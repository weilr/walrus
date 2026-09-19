"""Export a clear, traceable TRL2D evaluation workbook.

The workbook compares the base and fine-tuned Walrus runs while keeping the
validation statistics unchanged. Summary sheets reference detailed sheets
with Excel formulas; raw per-step curves live in one hidden chart-data sheet.

Example::

    python scripts/export_excel.py runs/eval_trl2d runs/eval_trl2d_ft \
        --labels base finetuned --out runs/walrus_trl2d_results.xlsx
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
from typing import Any

from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.chart.axis import ChartLines
from openpyxl.chart.marker import Marker
from openpyxl.chart.series import SeriesLabel
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.chart.text import RichText
from openpyxl.comments import Comment
from openpyxl.drawing.line import LineProperties
from openpyxl.drawing.text import CharacterProperties, Paragraph, ParagraphProperties
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

KEY_RE = re.compile(
    r"^(?P<dataset>[^/]+)/(?P<field>.+)_(?P<metric>[A-Za-z0-9]+)_T=(?P<time>[^_]+)_(?P<agg>[a-z]+)$"
)
CURVE_RE = re.compile(
    r"^(?P<field>.+)_(?P<metric>[A-Za-z0-9]+)_rollout_(?P<agg>[a-z]+)$"
)

METRICS = ["VRMSE", "NRMSE", "RelL2", "NMAE", "RelL1"]
FIELD_METRICS = ["NMAE", "NRMSE", "VRMSE"]
FIELDS = ["density", "pressure", "velocity_x", "velocity_y"]
AGGS = ["median", "mean"]
WINDOWS = ["1:1", "1:20", "21:60"]
PAPER = {"1:1": 0.0831, "1:20": 0.3393, "21:60": 0.8648}
TIME_LABELS = {
    "0:1": "第1步",
    "1:3": "第2–3步",
    "3:9": "第4–9步",
    "9:30": "第10–30步",
    "30:95": "第31–95步",
    "all": "全程（第1–95步）",
}
SHEET_ORDER = [
    "结果总览",
    "论文窗口",
    "一步预测",
    "Rollout分段",
    "逐字段",
    "趋势图",
    "参数定义",
    "逐步数据",
]

# Hover notes attached to table header cells (exact header -> note); labels of the two runs are added in main().
HEADER_NOTES: dict[str, str] = {
    "指标": "误差指标。NMAE/NRMSE/VRMSE 逐字段计算后对四个字段取平均；RelL1/RelL2 把四个字段与所有空间点展平后整体计算。公式见“参数定义”页。",
    "split": "test / valid：The Well 的测试集 / 验证集，各 9 条轨迹、每条 101 帧。",
    "统计": "对样本的聚合方式：median = 中位数，mean = 均值。一步预测对 855 个滑窗样本聚合；rollout 对 9 条轨迹聚合。",
    "提升率": "(Base − Fine-tuned) / Base。正值 = 微调后误差下降（绿色），负值 = 退化（红色）。",
    "一步 Base": "一步预测（上下文 6 帧预测第 7 帧），test 集 855 个样本的中位数，预训练 base 模型。",
    "一步 FT": "一步预测（上下文 6 帧预测第 7 帧），test 集 855 个样本的中位数，TRL2D 微调版。",
    "时间范围": "自回归 rollout 的步数区间（第 1 步 = 上下文之后的第一个预测帧）。",
    "原始索引": "loss dict 里的原始键：0-indexed、左闭右开，a:b 对应第 a+1 到第 b 步；all = 全程 95 步。",
    "字段": "物理场：density 密度、pressure 压力、velocity_x / velocity_y 速度分量。",
    "模式": "一步预测 = 每个滑窗只预测下一帧；Rollout全程 = 自回归 95 步后对整条轨迹取时间平均。",
    "step": "rollout 步数，1 = 上下文之后的第一个预测帧。",
    "论文参考": "arXiv 2511.15684 Tables 13–15 中 Walrus（TRL2D 微调，+500K 样本）的 median VRMSE。",
}
WINDOW_NOTES: dict[str, str] = {
    "1": "rollout 第 1 步（每步先对 9 条轨迹取中位数）。",
    "1:1": "rollout 第 1 步（每步先对 9 条轨迹取中位数）。",
    "1:20": "rollout 第 1–20 步：每步先对 9 条轨迹取中位数，再对 20 步取平均。对应论文 T[1:20]。",
    "21:60": "rollout 第 21–60 步：每步先对 9 条轨迹取中位数，再对 40 步取平均。对应论文 T[21:60]。",
}


def note_for(header: str) -> str | None:
    if header in HEADER_NOTES:
        return HEADER_NOTES[header]
    if "提升率" in header:
        return HEADER_NOTES["提升率"]
    match = re.match(r"T\[([^\]]+)\]\s*(.*)", header)
    if match:
        note = WINDOW_NOTES.get(match.group(1), "")
        who = HEADER_NOTES.get(match.group(2).strip(), "")
        return (note + (" " + who if who else "")) or None
    return None


NAVY = "17324D"
BLUE = "2A78D6"
GREEN = "1BAF7A"
LIGHT_BLUE = "EAF2FB"
LIGHT_GREEN = "E8F5EF"
LIGHT_RED = "FDECEC"
LIGHT_GRAY = "F4F6F8"
MID_GRAY = "667788"
WHITE = "FFFFFF"


def load_loss_dicts(run: str) -> dict[str, dict[tuple, float]]:
    """Load scalar validation summaries keyed by field, metric, time and aggregation."""
    out: dict[str, dict[tuple, float]] = {}
    paths = sorted(
        glob.glob(
            os.path.join(run, "viz", "loss_dicts", "*_loss_dict_epoch*_rank0.pkl")
        )
    )
    for path in paths:
        split = os.path.basename(path).split("_loss_dict_")[0]
        with open(path, "rb") as f:
            saved = pickle.load(f)
        table: dict[tuple, float] = {}
        for key, value in saved.items():
            if not key.startswith(split + "_"):
                continue
            match = KEY_RE.match(key[len(split) + 1 :])
            if match:
                table[
                    (
                        match["field"],
                        match["metric"],
                        match["time"],
                        match["agg"],
                    )
                ] = float(value)
        out[split] = table
    if not out:
        raise SystemExit(f"no loss_dict pickles under {run}")
    return out


def load_curves(run: str, split: str) -> dict[tuple, list[float]]:
    """Load per-step rollout curves keyed by field, metric and aggregation."""
    paths = sorted(
        glob.glob(
            os.path.join(
                run,
                "viz",
                "loss_dicts",
                f"rollout_{split}_time_logs_epoch*_rank0.pkl",
            )
        )
    )
    if not paths:
        raise SystemExit(f"no rollout_{split} time logs under {run}")
    with open(paths[-1], "rb") as f:
        logs = pickle.load(f)
    out: dict[tuple, list[float]] = {}
    for dataset_logs in logs.values():
        for key, tensor in dataset_logs.items():
            match = CURVE_RE.match(key.split("/", 1)[1])
            if match:
                out[(match["field"], match["metric"], match["agg"])] = [
                    float(value) for value in tensor
                ]
    return out


def time_sort_key(time_range: str) -> tuple[int, int]:
    return (1, 0) if time_range == "all" else (0, int(time_range.split(":")[0]))


def set_sheet_title(ws, title: str, subtitle: str, last_col: int) -> None:
    """Apply the shared page title and print setup."""
    end = get_column_letter(last_col)
    ws.sheet_view.showGridLines = False
    ws.merge_cells(f"A1:{end}1")
    ws.merge_cells(f"A2:{end}2")
    ws["A1"] = title
    ws["A2"] = subtitle
    ws["A1"].font = Font(name="Aptos Display", size=18, bold=True, color=WHITE)
    ws["A1"].fill = PatternFill("solid", fgColor=NAVY)
    ws["A1"].alignment = Alignment(vertical="center")
    ws["A2"].font = Font(name="Aptos", size=10, color=MID_GRAY)
    ws["A2"].fill = PatternFill("solid", fgColor=LIGHT_BLUE)
    ws["A2"].alignment = Alignment(vertical="center")
    ws.row_dimensions[1].height = 30
    ws.row_dimensions[2].height = 22
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0


def write_section_label(ws, row: int, text: str, last_col: int) -> None:
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=last_col)
    cell = ws.cell(row=row, column=1, value=text)
    cell.font = Font(name="Aptos", size=11, bold=True, color=NAVY)
    cell.fill = PatternFill("solid", fgColor=LIGHT_BLUE)
    cell.alignment = Alignment(vertical="center")
    ws.row_dimensions[row].height = 22


def write_table(
    ws,
    headers: list[str],
    rows: list[list[Any]],
    start_row: int,
    table_name: str,
    widths: list[float],
) -> int:
    """Write a styled, filterable table and return its final row."""
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=start_row, column=col, value=header)
        cell.font = Font(name="Aptos", bold=True, color=WHITE)
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.alignment = Alignment(
            horizontal="center", vertical="center", wrap_text=True
        )
        note = note_for(header)
        if note:  # definition on hover
            cell.comment = Comment(note, "walrus eval")
            cell.comment.width = 380
            cell.comment.height = 120
    ws.row_dimensions[start_row].height = 30

    for row_index, values in enumerate(rows, start_row + 1):
        for col, value in enumerate(values, 1):
            cell = ws.cell(row=row_index, column=col, value=value)
            cell.font = Font(name="Aptos", size=10)
            is_number = isinstance(value, float) or (
                isinstance(value, str) and value.startswith("=")
            )
            cell.alignment = Alignment(
                horizontal="right" if is_number else "left",
                vertical="center",
            )
            if isinstance(value, float):
                cell.number_format = "0.0000"
            if isinstance(value, str) and value.startswith("="):
                cell.number_format = "0.0000"
            if "提升" in headers[col - 1]:
                cell.number_format = "0.0%"

    end_row = start_row + len(rows)
    end_col = get_column_letter(len(headers))
    table = Table(displayName=table_name, ref=f"A{start_row}:{end_col}{end_row}")
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    ws.add_table(table)
    for col, width in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.freeze_panes = f"A{start_row + 1}"
    return end_row


def add_improvement_rules(ws, ranges: list[str]) -> None:
    green_fill = PatternFill("solid", fgColor=LIGHT_GREEN)
    red_fill = PatternFill("solid", fgColor=LIGHT_RED)
    green_font = Font(color="13754A")
    red_font = Font(color="B42318")
    for cell_range in ranges:
        ws.conditional_formatting.add(
            cell_range,
            CellIsRule(
                operator="greaterThan",
                formula=["0"],
                fill=green_fill,
                font=green_font,
            ),
        )
        ws.conditional_formatting.add(
            cell_range,
            CellIsRule(
                operator="lessThan",
                formula=["0"],
                fill=red_fill,
                font=red_font,
            ),
        )


def improvement_formula(base_cell: str, finetuned_cell: str) -> str:
    return f'=IFERROR(({base_cell}-{finetuned_cell})/{base_cell},"")'


def build_step_data(
    ws, labels: list[str], curves: dict, n_steps: int
) -> dict[tuple, int]:
    set_sheet_title(
        ws,
        "逐步统计数据（图表源）",
        "rollout_test；每一步对9条轨迹分别做 median / mean。此表供公式与图表追溯。",
        39,
    )
    col_specs: list[tuple[str, str, str, str]] = []
    for metric in METRICS:
        for agg in AGGS:
            for label in labels:
                col_specs.append((label, "full", metric, agg))
    for label in labels:
        for metric in ["NRMSE", "VRMSE"]:
            for field in FIELDS:
                col_specs.append((label, field, metric, "median"))

    headers = ["step"] + [
        f"{label} | {field} | {metric} | {agg}"
        for label, field, metric, agg in col_specs
    ]
    rows = [
        [step + 1]
        + [
            curves[label]["test"][(field, metric, agg)][step]
            for label, field, metric, agg in col_specs
        ]
        for step in range(n_steps)
    ]
    write_table(
        ws,
        headers,
        rows,
        start_row=4,
        table_name="StepDataTable",
        widths=[8] + [24] * len(col_specs),
    )
    return {spec: index + 2 for index, spec in enumerate(col_specs)}


def build_one_step(ws, labels: list[str], loss: dict) -> dict[tuple, str]:
    set_sheet_title(
        ws,
        "一步预测",
        "test / valid；每个 split 含855个滑窗样本。full 指标；误差越低越好。",
        6,
    )
    rows: list[list[Any]] = []
    refs: dict[tuple, str] = {}
    for split in ["test", "valid"]:
        for agg in AGGS:
            for metric in METRICS:
                row_number = 5 + len(rows)
                base_value = loss[labels[0]][split][("full", metric, "all", agg)]
                fine_value = loss[labels[1]][split][("full", metric, "all", agg)]
                rows.append(
                    [
                        split,
                        agg,
                        metric,
                        base_value,
                        fine_value,
                        improvement_formula(f"D{row_number}", f"E{row_number}"),
                    ]
                )
                refs[(split, agg, metric, labels[0])] = f"D{row_number}"
                refs[(split, agg, metric, labels[1])] = f"E{row_number}"
    end_row = write_table(
        ws,
        ["split", "统计", "指标", labels[0], labels[1], "提升率↓"],
        rows,
        start_row=4,
        table_name="OneStepTable",
        widths=[12, 12, 14, 14, 14, 14],
    )
    add_improvement_rules(ws, [f"F5:F{end_row}"])
    return refs


def build_paper_windows(
    ws,
    labels: list[str],
    step_columns: dict[tuple, int],
) -> dict[tuple, str]:
    set_sheet_title(
        ws,
        "论文窗口对比",
        "test / full / median；先逐步对9条轨迹取中位数，再在指定窗口内取平均。",
        15,
    )
    rows: list[list[Any]] = []
    refs: dict[tuple, str] = {}
    windows = {"1:1": (5, 5), "1:20": (5, 24), "21:60": (25, 64)}
    value_columns = {
        "1:1": ("B", "C", "D"),
        "1:20": ("E", "F", "G"),
        "21:60": ("H", "I", "J"),
    }
    for metric in METRICS:
        row_number = 5 + len(rows)
        values: list[Any] = [metric]
        for window in WINDOWS:
            start, end = windows[window]
            formulas = []
            for label in labels:
                source_col = get_column_letter(
                    step_columns[(label, "full", metric, "median")]
                )
                if start == end:
                    formula = f"='逐步数据'!{source_col}{start}"
                else:
                    formula = (
                        f"=AVERAGE('逐步数据'!{source_col}{start}:{source_col}{end})"
                    )
                formulas.append(formula)
            base_col, fine_col, _ = value_columns[window]
            values.extend(
                [
                    formulas[0],
                    formulas[1],
                    improvement_formula(
                        f"{base_col}{row_number}", f"{fine_col}{row_number}"
                    ),
                ]
            )
            refs[(metric, window, labels[0])] = f"{base_col}{row_number}"
            refs[(metric, window, labels[1])] = f"{fine_col}{row_number}"
        rows.append(values)

    headers = ["指标"]
    for window in (
        WINDOWS
    ):  # Excel tables need unique column names, so tag each 提升率 with its window
        headers.extend(
            [
                f"T[{window}] {labels[0]}",
                f"T[{window}] {labels[1]}",
                f"T[{window}] 提升率",
            ]
        )
    end_row = write_table(
        ws,
        headers,
        rows,
        start_row=4,
        table_name="PaperWindowTable",
        widths=[14] + [15] * 9,
    )
    add_improvement_rules(
        ws,
        [f"D5:D{end_row}", f"G5:G{end_row}", f"J5:J{end_row}"],
    )

    reference_headers = ["论文参考", "T[1:1]", "T[1:20]", "T[21:60]"]
    for col, value in enumerate(reference_headers, 12):
        cell = ws.cell(row=4, column=col, value=value)
        cell.font = Font(name="Aptos", bold=True, color=WHITE)
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        note = note_for(value)
        if note:
            cell.comment = Comment(note, "walrus eval")
            cell.comment.width, cell.comment.height = 380, 120
    ws["L5"] = "Walrus-ft VRMSE (median)"
    ws["M5"] = PAPER["1:1"]
    ws["N5"] = PAPER["1:20"]
    ws["O5"] = PAPER["21:60"]
    for cell in ws[5][11:15]:
        cell.font = Font(name="Aptos", size=10)
        cell.alignment = Alignment(
            horizontal="right" if isinstance(cell.value, float) else "left"
        )
        if isinstance(cell.value, float):
            cell.number_format = "0.0000"
    ws["L6"] = "本地 finetuned 与论文之差"
    for col, window in zip("MNO", WINDOWS):
        base_col, fine_col, _ = value_columns[window]
        row_vrmse = 5 + METRICS.index("VRMSE")
        ws[f"{col}6"] = f"={fine_col}{row_vrmse}-{col}5"
        ws[f"{col}6"].number_format = "+0.0000;-0.0000;0.0000"
        ws[f"{col}6"].font = Font(name="Aptos", size=10)
        ws[f"{col}6"].alignment = Alignment(horizontal="right")
    ws["L6"].font = Font(name="Aptos", size=10)
    ws["L7"] = "来源"
    ws["M7"] = (
        "arXiv 2511.15684, Tables 13–15"  # details in the 论文参考 header note and 参数定义 sheet
    )
    ws.merge_cells("M7:O7")
    for ref in ("L7", "M7"):
        ws[ref].font = Font(name="Aptos", size=10, color=MID_GRAY)
    ws.column_dimensions["L"].width = 27
    for col in range(13, 16):
        ws.column_dimensions[get_column_letter(col)].width = 14
    return refs


def build_dashboard(
    ws,
    labels: list[str],
    one_step_refs: dict[tuple, str],
    paper_refs: dict[tuple, str],
) -> None:
    set_sheet_title(
        ws,
        "Walrus TRL2D 复现结果总览",
        "Base vs Fine-tuned｜test split｜full 指标｜median｜所有误差均为比例，越低越好",
        13,
    )
    write_section_label(ws, 4, "核心结果", 13)
    headers = [  # column names must be unique inside an Excel table
        "指标",
        "一步 Base",
        "一步 FT",
        "一步 提升率",
        "T[1] Base",
        "T[1] FT",
        "T[1] 提升率",
        "T[1:20] Base",
        "T[1:20] FT",
        "T[1:20] 提升率",
        "T[21:60] Base",
        "T[21:60] FT",
        "T[21:60] 提升率",
    ]
    rows: list[list[Any]] = []
    for metric in METRICS:
        row_number = 6 + len(rows)
        one_base = one_step_refs[("test", "median", metric, labels[0])]
        one_fine = one_step_refs[("test", "median", metric, labels[1])]
        row: list[Any] = [
            metric,
            f"='一步预测'!{one_base}",
            f"='一步预测'!{one_fine}",
            improvement_formula(f"B{row_number}", f"C{row_number}"),
        ]
        for window, base_col, fine_col in [
            ("1:1", "E", "F"),
            ("1:20", "H", "I"),
            ("21:60", "K", "L"),
        ]:
            paper_base = paper_refs[(metric, window, labels[0])]
            paper_fine = paper_refs[(metric, window, labels[1])]
            row.extend(
                [
                    f"='论文窗口'!{paper_base}",
                    f"='论文窗口'!{paper_fine}",
                    improvement_formula(
                        f"{base_col}{row_number}", f"{fine_col}{row_number}"
                    ),
                ]
            )
        rows.append(row)
    end_row = write_table(
        ws,
        headers,
        rows,
        start_row=5,
        table_name="DashboardTable",
        widths=[12] + [14] * 12,
    )
    add_improvement_rules(
        ws,
        [
            f"D6:D{end_row}",
            f"G6:G{end_row}",
            f"J6:J{end_row}",
            f"M6:M{end_row}",
        ],
    )

    write_section_label(ws, 12, "T[1:20] 关键指标", 13)
    card_specs = [(1, "VRMSE", 6), (5, "RelL2", 8), (9, "RelL1", 10)]
    for start_col, metric, source_row in card_specs:
        end_col = start_col + 3
        ws.merge_cells(
            start_row=13,
            start_column=start_col,
            end_row=13,
            end_column=end_col,
        )
        title_cell = ws.cell(row=13, column=start_col, value=metric)
        title_cell.font = Font(name="Aptos", size=12, bold=True, color=WHITE)
        title_cell.fill = PatternFill("solid", fgColor=NAVY)
        title_cell.alignment = Alignment(horizontal="center")
        labels_row = [labels[0], labels[1], "提升", "方向"]
        values_row = [
            f"=H{source_row}",
            f"=I{source_row}",
            f"=J{source_row}",
            "越低越好",
        ]
        for offset, label in enumerate(labels_row):
            cell = ws.cell(row=14, column=start_col + offset, value=label)
            cell.font = Font(name="Aptos", bold=True, color=MID_GRAY)
            cell.fill = PatternFill("solid", fgColor=LIGHT_GRAY)
            cell.alignment = Alignment(horizontal="center")
        for offset, value in enumerate(values_row):
            cell = ws.cell(row=15, column=start_col + offset, value=value)
            cell.font = Font(name="Aptos", size=11)
            cell.alignment = Alignment(horizontal="center")
            if offset < 2:
                cell.number_format = "0.0000"
            elif offset == 2:
                cell.number_format = "0.0%"
                cell.fill = PatternFill("solid", fgColor=LIGHT_GREEN)
                cell.font = Font(name="Aptos", bold=True, color="13754A")

    ws.merge_cells("A18:M18")
    ws["A18"] = (
        "读表顺序：先看提升率，再看 T[1:20] 与 T[21:60]；一步预测用于短期精度，"
        "长期窗口用于判断自回归稳定性。表头悬停可查看每列定义，完整定义见“参数定义”页。"
    )
    ws["A18"].font = Font(name="Aptos", italic=True, color=MID_GRAY)
    ws["A18"].alignment = Alignment(wrap_text=True)
    ws.row_dimensions[18].height = 30

    write_section_label(
        ws, 20, "指标速查（x = 预测，y = 真值；均为无量纲比例，越低越好）", 13
    )
    quick = [
        (
            "VRMSE",
            "RMSE(x−y) / std(y)，逐字段计算后取四字段平均；论文主指标，对常数偏移敏感",
        ),
        ("NRMSE", "‖x−y‖₂ / (‖y‖₂+ε)，逐字段相对 L2 误差，取四字段平均"),
        ("NMAE", "Σ|x−y| / (Σ|y|+ε)，逐字段相对 L1 误差，取四字段平均"),
        (
            "RelL2",
            "‖x−y‖₂ / (‖y‖₂+ε)，四个字段与全部空间点展平为一个向量后计算（常用的 rel L2）",
        ),
        ("RelL1", "Σ|x−y| / (Σ|y|+ε)，同上展平后计算（常用的 rel L1）"),
        ("提升率", "(Base − Fine-tuned) / Base，正 = 改善"),
        (
            "T[a:b]",
            "rollout 第 a–b 步：每步先对 9 条轨迹取中位数，再在窗口内取平均；一步 = test 集 855 个滑窗样本的中位数",
        ),
    ]
    for offset, (name, text) in enumerate(quick):
        row = 21 + offset
        name_cell = ws.cell(row=row, column=1, value=name)
        name_cell.font = Font(name="Aptos", bold=True, color=NAVY)
        name_cell.fill = PatternFill("solid", fgColor=LIGHT_GRAY)
        name_cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=13)
        text_cell = ws.cell(row=row, column=2, value=text)
        text_cell.font = Font(name="Aptos", size=10)
        text_cell.alignment = Alignment(vertical="center", wrap_text=True)
        ws.row_dimensions[row].height = 20
    ws.freeze_panes = "B6"


def build_rollout_segments(ws, labels: list[str], loss: dict) -> None:
    set_sheet_title(
        ws,
        "Rollout 分段统计",
        "每条轨迹先在时间段内取平均，再对9条轨迹做 median / mean；full 指标。",
        8,
    )
    rows: list[list[Any]] = []
    for split in ["rollout_test", "rollout_valid"]:
        times = sorted({key[2] for key in loss[labels[0]][split]}, key=time_sort_key)
        for time_range in times:
            for agg in AGGS:
                for metric in METRICS:
                    row_number = 5 + len(rows)
                    base_value = loss[labels[0]][split][
                        ("full", metric, time_range, agg)
                    ]
                    fine_value = loss[labels[1]][split][
                        ("full", metric, time_range, agg)
                    ]
                    rows.append(
                        [
                            split.removeprefix("rollout_"),
                            TIME_LABELS.get(time_range, time_range),
                            time_range,
                            agg,
                            metric,
                            base_value,
                            fine_value,
                            improvement_formula(f"F{row_number}", f"G{row_number}"),
                        ]
                    )
    end_row = write_table(
        ws,
        [
            "split",
            "时间范围",
            "原始索引",
            "统计",
            "指标",
            labels[0],
            labels[1],
            "提升率↓",
        ],
        rows,
        start_row=4,
        table_name="RolloutSegmentTable",
        widths=[11, 20, 13, 11, 13, 14, 14, 14],
    )
    add_improvement_rules(ws, [f"H5:H{end_row}"])


def build_fields(ws, labels: list[str], loss: dict) -> None:
    set_sheet_title(
        ws,
        "逐字段误差",
        "仅展示真正逐字段计算的 NMAE / NRMSE / VRMSE；全局 RelL1 / RelL2 请查看总览。",
        8,
    )
    rows: list[list[Any]] = []
    modes = [
        ("一步预测", ["test", "valid"]),
        ("Rollout全程", ["rollout_test", "rollout_valid"]),
    ]
    for mode, splits in modes:
        for split in splits:
            for agg in AGGS:
                for field in FIELDS:
                    for metric in FIELD_METRICS:
                        row_number = 5 + len(rows)
                        base_value = loss[labels[0]][split][(field, metric, "all", agg)]
                        fine_value = loss[labels[1]][split][(field, metric, "all", agg)]
                        rows.append(
                            [
                                mode,
                                split.removeprefix("rollout_"),
                                agg,
                                field,
                                metric,
                                base_value,
                                fine_value,
                                improvement_formula(f"F{row_number}", f"G{row_number}"),
                            ]
                        )
    end_row = write_table(
        ws,
        ["模式", "split", "统计", "字段", "指标", labels[0], labels[1], "提升率↓"],
        rows,
        start_row=4,
        table_name="FieldMetricTable",
        widths=[15, 11, 11, 16, 13, 14, 14, 14],
    )
    add_improvement_rules(ws, [f"H5:H{end_row}"])


def build_line_chart(
    source_ws,
    columns: list[int],
    labels: list[str],
    n_steps: int,
    title: str,
) -> LineChart:
    chart = LineChart()
    chart.title = title
    chart.style = 2
    chart.height = 8.5
    chart.width = 16
    # no x-axis title: Excel renders it on top of the tick labels; the chart title names the axis instead
    chart.x_axis.tickLblSkip = 10
    chart.x_axis.tickMarkSkip = 10
    chart.x_axis.tickLblPos = "low"
    chart.x_axis.crosses = "min"
    chart.y_axis.scaling.logBase = 10
    chart.y_axis.number_format = "General"  # log ticks: 0.001, 0.01, 0.1, 1, 10
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    chart.legend.position = "r"
    chart.visible_cells_only = False
    # light grid on both axes; minor gridlines give the log y axis a finer scale (2..9 in each decade)
    major = ChartLines(spPr=GraphicalProperties(ln=LineProperties(solidFill="D9E0E8", w=6350)))
    minor = ChartLines(spPr=GraphicalProperties(ln=LineProperties(solidFill="EEF2F6", w=6350)))
    chart.y_axis.majorGridlines = major
    chart.y_axis.minorGridlines = minor
    chart.x_axis.majorGridlines = ChartLines(
        spPr=GraphicalProperties(ln=LineProperties(solidFill="D9E0E8", w=6350))
    )
    # thin, light axis lines and small grey tick labels
    for axis in (chart.x_axis, chart.y_axis):
        axis.spPr = GraphicalProperties(ln=LineProperties(solidFill="B8C2CC", w=6350))  # 0.5 pt
        label_font = CharacterProperties(sz=800, solidFill=MID_GRAY)  # 8 pt
        axis.txPr = RichText(
            p=[Paragraph(pPr=ParagraphProperties(defRPr=label_font), endParaRPr=label_font)]
        )
    chart.y_axis.majorTickMark = "out"
    chart.y_axis.minorTickMark = "out"
    for column in columns:
        chart.add_data(
            Reference(
                source_ws,
                min_col=column,
                min_row=4,
                max_row=n_steps + 4,
            ),
            titles_from_data=True,
        )
    chart.set_categories(
        Reference(source_ws, min_col=1, min_row=5, max_row=n_steps + 4)
    )
    for index, (series, label) in enumerate(zip(chart.series, labels)):
        color = [BLUE, GREEN][index]
        series.tx = SeriesLabel(v=label)
        series.graphicalProperties.line.solidFill = color
        series.graphicalProperties.line.width = 15875  # 1.25 pt
        # One dot per actual data point. Excel snaps the marker box to whole pixels; measured on this
        # machine a 4 pt circle lands ~1-2 px up-left of the line vertex while 5 pt is within 0.3 px.
        series.marker = Marker(symbol="circle", size=5)
        series.marker.graphicalProperties = GraphicalProperties(
            solidFill=color, ln=LineProperties(noFill=True)
        )
        series.smooth = False
    return chart


def build_trends(
    ws,
    source_ws,
    labels: list[str],
    step_columns: dict[tuple, int],
    n_steps: int,
) -> None:
    set_sheet_title(
        ws,
        "Rollout 趋势",
        "rollout_test；每一步对9条轨迹取 median；纵轴为对数刻度。",
        17,
    )
    anchors = ["A4", "J4", "A23", "J23"]
    titles = {
        "VRMSE": "VRMSE 随 rollout 步数变化（论文指标；对数纵轴）",
        "NRMSE": "NRMSE 随 rollout 步数变化（逐字段相对 L2 的平均；对数纵轴）",
        "RelL2": "RelL2 随 rollout 步数变化（所有字段展平；对数纵轴）",
        "RelL1": "RelL1 随 rollout 步数变化（所有字段展平；对数纵轴）",
    }
    for metric, anchor in zip(["VRMSE", "NRMSE", "RelL2", "RelL1"], anchors):
        columns = [step_columns[(label, "full", metric, "median")] for label in labels]
        chart = build_line_chart(source_ws, columns, labels, n_steps, titles[metric])
        ws.add_chart(chart, anchor)
    for col in range(1, 18):
        ws.column_dimensions[get_column_letter(col)].width = 11


def build_methods(ws, labels: list[str], runs: dict[str, str]) -> None:
    """Definitions of every dataset / model / evaluation parameter and metric used in the workbook."""
    set_sheet_title(
        ws,
        "参数与指标定义",
        "数据集、模型、微调改动、评测配置、指标公式、统计口径与论文参考；表头悬停注释与此页一致。",
        3,
    )
    rows = [
        # ---- 数据集
        [
            "数据集",
            "名称",
            "turbulent_radiative_layer_2D（The Well，PolymathicAI）：二维湍流辐射层，冷热气体界面的热不稳定性演化；控制参数 tcool（冷却时间）。",
        ],
        [
            "数据集",
            "网格与边界",
            "128 × 384 笛卡尔网格，x 方向周期边界、y 方向开放边界；二维（n_spatial_dims = 2）。",
        ],
        [
            "数据集",
            "字段",
            "density（密度，标量）、pressure（压力，标量）、velocity_x / velocity_y（速度矢量的两个分量）；共 4 个通道。",
        ],
        [
            "数据集",
            "切分",
            "train / valid / test 各 9 条轨迹，每条 101 帧。本评测使用 valid 与 test；train 未参与。",
        ],
        [
            "数据集",
            "一步预测样本",
            "上下文 6 帧预测第 7 帧；每条轨迹 101 − 6 = 95 个滑窗，9 条轨迹共 855 个样本（test、valid 各 855）。",
        ],
        [
            "数据集",
            "Rollout 样本",
            "每条轨迹取前 6 帧作上下文，自回归预测 95 步直到轨迹末尾；共 9 条轨迹（test、valid 各 9 条）。",
        ],
        # ---- 模型
        [
            "模型",
            "Walrus",
            "PolymathicAI 发布的流体物理基础模型（IsotropicModel），约 1.28 B 参数，hidden_dim 1408，40 个 processor block；在 19 个 The Well 数据集上联合预训练，TRL2D 属于预训练集（非 OOD）。",
        ],
        ["模型", "n_steps_input = 6", "上下文长度：模型一次读入 6 帧，输出 1 帧。"],
        [
            "模型",
            "delta prediction",
            "增量预测：网络输出下一帧与当前帧之差，再加回当前帧。",
        ],
        [
            "模型",
            "patch jittering",
            "对 patch 分块网格做随机偏移，减少分块伪影（论文 Table 2 消融，TRL2D 上 0.75 vs 76.9）。",
        ],
        [
            "模型",
            "revin（归一化）",
            f"{labels[0]}：SamplewiseRevNormalization，按样本 RMS 归一化再反归一化；{labels[1]}：GlobalRevNormalization，用数据集全局 RMS（stats.yaml）归一化。",
        ],
        [
            "模型",
            f"{labels[0]} 权重",
            "HuggingFace polymathic-ai/walrus 的 walrus.pt（5.15 GB，857 个张量）。",
        ],
        [
            "模型",
            f"{labels[1]} 权重",
            "Flatiron walrus_project_checkpoints/turbulent_radiative_layer_2D/coalesced.pth（938 个张量）：在 TRL2D 上从 base 微调，论文 Sec 5.2 的 Walrus 列（+500K 样本、50 epoch、lr 1e-4、batch 1）。",
        ],
        # ---- 微调改动
        [
            "微调改动",
            "learnable_rope = True",
            "RoPE（旋转位置编码）的频率从固定值改为可学习参数。",
        ],
        [
            "微调改动",
            "rope_per_axis = True",
            "每个空间轴使用独立的 RoPE 频率，权重中每层出现 freqs.0 / freqs.1 / freqs.2。",
        ],
        [
            "微调改动",
            "ape_shape = [32, 33, 1]",
            "新增可学习的绝对位置编码 ape，形状 1×1×1408×32×33×1（patch 网格尺寸）。",
        ],
        [
            "微调改动",
            "load_chkpt_after_finetuning_expansion = True",
            "先按上述改动扩展模型结构，再严格加载微调权重（否则键不匹配）。",
        ],
        # ---- 评测配置
        [
            "评测配置",
            "validation_mode = True",
            "只运行 valid / test 的一步预测与 rollout，不训练。",
        ],
        [
            "评测配置",
            "batch_size = 1",
            "单样本推理；显存约 5.5 GB，约 1 s / 样本（RTX 4070 SUPER 12 GB）。",
        ],
        [
            "评测配置",
            "enable_amp = False",
            "fp32 / TF32 推理；bf16 autocast 在该显卡上慢约 3 倍且更占显存，故关闭。",
        ],
        [
            "评测配置",
            "dt_stride = 1",
            "验证集帧间步长固定为 1（与训练配置 max_dt_stride 无关）。",
        ],
        ["评测配置", "max_rollout_steps", "配置为 200，实际受轨迹长度限制为 95 步。"],
        [
            "评测配置",
            "validation_suite",
            "NMAE、NRMSE、VRMSE（the_well.benchmark.metrics）+ RelL1、RelL2（walrus/rel_metrics.py）。",
        ],
        [
            "评测配置",
            "batch_aggregation_fns",
            "torch.mean、torch.median、torch.std；本工作簿列出 mean 与 median。",
        ],
        # ---- 指标
        [
            "指标",
            "记号",
            "x = 预测，y = 真值；每个样本、每个时间步先在空间维上计算，再对样本聚合。当前配置未覆盖 validation_epsilon，因此实际使用 Trainer 默认 ε = 1e-5 防止除零。所有指标无量纲，0.1000 = 10%。",
        ],
        [
            "指标",
            "NMAE",
            "Σ|x − y| / (Σ|y| + ε)：逐字段相对 L1 误差（the_well NMAE）。",
        ],
        [
            "指标",
            "NRMSE",
            "‖x − y‖₂ / (‖y‖₂ + ε)：逐字段相对 L2 误差（the_well NRMSE）。",
        ],
        [
            "指标",
            "VRMSE",
            "RMSE(x − y) / (std(y) + ε)：用真值标准差归一化的 RMSE，论文主指标。分母只含波动、不含均值，因此对常数偏移非常敏感（pressure 的 VRMSE 远大于其 NRMSE）。",
        ],
        [
            "指标",
            "RelL1",
            "Σ|x − y| / (Σ|y| + ε)，求和范围为四个字段 × 全部空间点（展平为一个向量）：神经算子文献中常用的 rel L1。",
        ],
        [
            "指标",
            "RelL2",
            "‖x − y‖₂ / (‖y‖₂ + ε)，同上展平：常用的 rel L2。量级由数值最大的 density 主导，对小幅值字段不敏感。",
        ],
        [
            "指标",
            "full",
            "NMAE / NRMSE / VRMSE 的 full = 四个字段的算术平均；RelL1 / RelL2 的 full = 展平值本身（各字段列相同）。",
        ],
        # ---- 统计口径
        [
            "统计口径",
            "mean / median",
            "一步预测：对 855 个滑窗样本聚合。Rollout 分段：每条轨迹先在时间段内取平均，再对 9 条轨迹聚合。",
        ],
        [
            "统计口径",
            "分段原始索引 a:b",
            "0-indexed、左闭右开，对应第 a+1 到第 b 步：0:1 = 第 1 步，1:3 = 第 2–3 步，3:9 = 第 4–9 步，9:30 = 第 10–30 步，30:95 = 第 31–95 步，all = 全程。",
        ],
        [
            "统计口径",
            "论文窗口 T[1:1] / T[1:20] / T[21:60]",
            "每步先对 9 条轨迹取中位数，再在窗口内取平均（步数从 1 起算）。论文未写明聚合顺序，这是与论文数值最接近的口径。",
        ],
        [
            "统计口径",
            "提升率",
            "(Base − Fine-tuned) / Base：正值 = 微调后误差下降（绿色），负值 = 退化（红色）。",
        ],
        # ---- 论文参考
        [
            "论文参考",
            "来源",
            "Walrus 论文 arXiv 2511.15684，Tables 13–15 的 TRL (2D) 列：one-step 0.0831 ± 0.0638，T[1:20] 0.3393，T[21:60] 0.8648（median VRMSE，test）。",
        ],
        [
            "论文参考",
            "口径差异",
            "论文的 one-step 更接近本表“一步预测 median”（855 样本），而非 rollout 第 1 步；论文对比基线 MPP-AViT-L / Poseidon-L / DPOT-H 使用 4.5M 样本微调。",
        ],
        # ---- 运行
        ["运行", f"{labels[0]} 运行目录", runs[labels[0]]],
        ["运行", f"{labels[1]} 运行目录", runs[labels[1]]],
        [
            "运行",
            "生成脚本",
            "scripts/export_excel.py（本工作簿）；scripts/summarize_eval.py、scripts/paper_windows.py 为控制台汇总。",
        ],
        [
            "运行",
            "工作表顺序",
            "结果总览 → 论文窗口 → 一步预测 → Rollout分段 → 逐字段 → 趋势图 → 参数定义；“逐步数据”为隐藏的图表数据源。",
        ],
    ]
    end_row = write_table(
        ws,
        ["类别", "参数 / 术语", "定义"],
        rows,
        start_row=4,
        table_name="DefinitionTable",
        widths=[12, 34, 110],
    )
    for row in range(5, end_row + 1):
        ws.cell(row=row, column=1).alignment = Alignment(
            vertical="top", horizontal="center"
        )
        ws.cell(row=row, column=2).alignment = Alignment(vertical="top", wrap_text=True)
        ws.cell(row=row, column=3).alignment = Alignment(vertical="top", wrap_text=True)
        text = str(ws.cell(row=row, column=3).value)
        units = sum(
            2 if ord(ch) > 0x2E80 else 1 for ch in text
        )  # CJK glyphs are about twice as wide
        ws.row_dimensions[row].height = 15 * max(1, -(-units // 140))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs=2)
    parser.add_argument("--labels", nargs=2, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runs = dict(zip(args.labels, args.run_dirs))
    HEADER_NOTES[args.labels[0]] = (
        "预训练 Walrus（base，HF polymathic-ai/walrus）的误差。"
    )
    HEADER_NOTES[args.labels[1]] = (
        "TRL2D 微调版 Walrus（Flatiron coalesced.pth）的误差。"
    )
    loss = {label: load_loss_dicts(run) for label, run in runs.items()}
    curves = {
        label: {split: load_curves(run, split) for split in ["test", "valid"]}
        for label, run in runs.items()
    }
    n_steps = len(next(iter(curves[args.labels[0]]["test"].values())))

    workbook = Workbook()
    workbook.active.title = SHEET_ORDER[0]
    for sheet_name in SHEET_ORDER[1:]:
        workbook.create_sheet(sheet_name)

    step_columns = build_step_data(workbook["逐步数据"], args.labels, curves, n_steps)
    one_step_refs = build_one_step(workbook["一步预测"], args.labels, loss)
    paper_refs = build_paper_windows(workbook["论文窗口"], args.labels, step_columns)
    build_dashboard(workbook["结果总览"], args.labels, one_step_refs, paper_refs)
    build_rollout_segments(workbook["Rollout分段"], args.labels, loss)
    build_fields(workbook["逐字段"], args.labels, loss)
    build_trends(
        workbook["趋势图"],
        workbook["逐步数据"],
        args.labels,
        step_columns,
        n_steps,
    )
    build_methods(workbook["参数定义"], args.labels, runs)

    workbook["逐步数据"].sheet_state = "hidden"
    workbook.calculation.fullCalcOnLoad = True
    workbook.calculation.forceFullCalc = True
    workbook.calculation.calcMode = "auto"

    output = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    workbook.save(output)
    print(f"wrote {output}: sheets={workbook.sheetnames}, steps={n_steps}")


if __name__ == "__main__":
    main()
