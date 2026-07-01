from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("results/attack_centric_final")
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
AUDITS = ROOT / "audits"

REQUIRED_TABLES = [
    "b1_metric_trap_audit_all_datasets",
    "c1_ctt_all_settings_corrected_benchmark",
    "d1_table13_case_study",
    "e1_ranking_inversion_all",
    "f1_low_fpr_deployment",
    "g1_event_level_evaluation",
    "h1_grain_feature_granularity_analysis",
    "i1_external_corrected_sanity",
    "paper_table1_dataset_stats",
    "paper_table2_metric_trap_audit",
    "paper_table3_corrected_ctt_benchmark",
    "paper_table4_low_fpr",
    "paper_table5_event_level",
    "paper_table6_grain_ablation",
    "paper_table7_external_sanity",
]

REQUIRED_FIGS = [
    "paper_fig1_metric_trap_across_datasets.svg",
    "paper_fig2_ctt_corrected_benchmark.svg",
    "paper_fig3_table13_case_study.svg",
    "paper_fig4_ranking_inversion.svg",
    "paper_fig5_low_fpr_deployment.svg",
    "paper_fig6_event_level.svg",
    "paper_fig7_grain_feature_preservation.svg",
    "paper_fig8_external_sanity.svg",
    "paper_fig9_attack_centric_pipeline.svg",
]

REQUIRED_MD = [
    "final_paper_theme.md",
    "final_contributions.md",
    "recommended_paper_outline.md",
    "main_claims.md",
    "unsafe_claims_do_not_write.md",
    "threats_to_validity.md",
    "security4_readiness.md",
    "paper_figure_table_inventory.md",
]


def audit_csv(path: Path) -> dict:
    row = {"file_path": str(path), "kind": "csv", "exists": path.exists(), "file_size_bytes": path.stat().st_size if path.exists() else 0}
    if not path.exists():
        row.update({"readable": False, "num_rows": 0, "num_columns": 0, "is_empty": True, "status": "missing"})
        return row
    try:
        df = pd.read_csv(path)
        row.update({"readable": True, "num_rows": len(df), "num_columns": len(df.columns), "is_empty": len(df) == 0, "status": "ok" if len(df) else "empty"})
    except Exception as exc:
        row.update({"readable": False, "num_rows": 0, "num_columns": 0, "is_empty": True, "status": f"error:{exc}"})
    return row


def audit_text(path: Path, kind: str, marker: str | None = None) -> dict:
    row = {"file_path": str(path), "kind": kind, "exists": path.exists(), "file_size_bytes": path.stat().st_size if path.exists() else 0}
    if not path.exists():
        row.update({"readable": False, "num_rows": None, "num_columns": None, "is_empty": True, "status": "missing"})
        return row
    text = path.read_text(errors="ignore")
    ok_marker = marker in text if marker else True
    row.update({"readable": True, "num_rows": None, "num_columns": None, "is_empty": len(text.strip()) == 0, "status": "ok" if text.strip() and ok_marker else "invalid"})
    return row


def main() -> int:
    AUDITS.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in REQUIRED_TABLES:
        rows.append(audit_csv(TABLES / f"{name}.csv"))
        rows.append(audit_text(TABLES / f"{name}.tex", "tex"))
    for fig in REQUIRED_FIGS:
        rows.append(audit_text(FIGS / fig, "svg", "<svg"))
    for md in REQUIRED_MD:
        rows.append(audit_text(ROOT / md, "md"))
    df = pd.DataFrame(rows)
    df.to_csv(AUDITS / "output_integrity_audit.csv", index=False)
    failures = df[df["status"].ne("ok")]
    (AUDITS / "output_integrity_audit.md").write_text(
        "# Attack-Centric Final Output Integrity Audit\n\n"
        f"Checked files: {len(df)}\n\nFailures: {len(failures)}\n\n"
        + (f"```csv\n{failures.to_csv(index=False)}```\n" if len(failures) else "All required files are present, readable, and non-empty.\n"),
        encoding="utf-8",
    )
    return 0 if failures.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
