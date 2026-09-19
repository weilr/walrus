import subprocess
import sys
from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl", reason="Excel report tests require openpyxl")

ROOT = Path(__file__).resolve().parents[1]
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


@pytest.fixture(scope="module")
def clear_workbook(tmp_path_factory):
    run_dirs = [ROOT / "runs" / name for name in ("eval_trl2d", "eval_trl2d_ft")]
    required_patterns = [
        f"{split}_loss_dict_epoch*_rank0.pkl"
        for split in ("test", "valid", "rollout_test", "rollout_valid")
    ] + [
        f"rollout_{split}_time_logs_epoch*_rank0.pkl" for split in ("test", "valid")
    ]
    for run_dir in run_dirs:
        for pattern in required_patterns:
            if not any((run_dir / "viz" / "loss_dicts").glob(pattern)):
                pytest.skip(f"Requires local evaluation output: {run_dir} ({pattern})")

    output = tmp_path_factory.mktemp("export_excel") / "walrus_trl2d_results_v2.xlsx"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_excel.py"),
            *(str(run_dir) for run_dir in run_dirs),
            "--labels",
            "base",
            "finetuned",
            "--out",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )
    return openpyxl.load_workbook(output, data_only=False)


def test_exported_workbook_has_clear_information_hierarchy(clear_workbook):
    assert clear_workbook.sheetnames == SHEET_ORDER
    assert clear_workbook["逐步数据"].sheet_state == "hidden"
    assert clear_workbook["结果总览"]["A1"].value == "Walrus TRL2D 复现结果总览"


def test_dashboard_uses_traceable_formulas(clear_workbook):
    dashboard = clear_workbook["结果总览"]
    assert dashboard["B6"].value == "='一步预测'!D5"
    assert dashboard["C6"].value == "='一步预测'!E5"
    assert dashboard["D6"].value == '=IFERROR((B6-C6)/B6,"")'
    assert dashboard["B6"].number_format == "0.0000"

    paper = clear_workbook["论文窗口"]
    assert paper["B5"].value == "='逐步数据'!B5"
    assert paper["E5"].value == "=AVERAGE('逐步数据'!B5:B24)"
    assert paper["H5"].value == "=AVERAGE('逐步数据'!B25:B64)"
    assert paper["B5"].number_format == "0.0000"


def test_detailed_outputs_remove_repeated_global_metrics_and_keep_key_charts(
    clear_workbook,
):
    fields = clear_workbook["逐字段"]
    metrics = {
        fields.cell(row=row, column=5).value for row in range(5, fields.max_row + 1)
    }
    assert metrics == {"NMAE", "NRMSE", "VRMSE"}
    assert len(clear_workbook["趋势图"]._charts) == 4


def test_definition_sheet_records_actual_validation_epsilon(clear_workbook):
    definitions = clear_workbook["参数定义"]
    notation_rows = [
        row
        for row in range(5, definitions.max_row + 1)
        if definitions.cell(row=row, column=2).value == "记号"
    ]
    assert len(notation_rows) == 1
    assert "ε = 1e-5" in definitions.cell(row=notation_rows[0], column=3).value
