from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("results/final_grain_can")
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
AUDITS = ROOT / "audits"

REQUIRED_COLUMNS = {
    "a1_official_sample_negative_stability.csv": {"dataset", "model", "f1", "auroc", "aupr"},
    "b1_granularity_full_matrix.csv": {"dataset", "model", "granularity", "f1"},
    "c1_test04_candidate_leaderboard.csv": {"dataset", "model", "f1"},
    "e1_low_fpr_leaderboard.csv": {"dataset", "model", "fpr_budget"},
    "f2_event_level_metrics.csv": {"dataset", "model", "event_recall"},
}

EXPECTED_LEGEND = {
    "paper_fig2_granularity_comparison.svg",
    "paper_fig5_low_fpr_curves.svg",
}


def audit_csv(path: Path) -> dict:
    row = {
        "file_type": "csv",
        "file_path": str(path),
        "exists": path.exists(),
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
        "num_rows": 0,
        "num_columns": 0,
        "readable_by_pandas": False,
        "has_required_columns": False,
        "is_empty": True,
        "source_generation_script": "scripts/plot_final_grain_can.py or revision scripts",
    }
    if path.exists() and path.stat().st_size > 0:
        try:
            df = pd.read_csv(path)
            row["readable_by_pandas"] = True
            row["num_rows"] = int(len(df))
            row["num_columns"] = int(len(df.columns))
            row["is_empty"] = bool(df.empty or len(df.columns) == 0)
            req = REQUIRED_COLUMNS.get(path.name)
            row["has_required_columns"] = True if req is None else req.issubset(set(df.columns))
        except Exception as exc:
            row["error"] = repr(exc)
    return row


def audit_tex(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    return {
        "file_type": "tex",
        "file_path": str(path),
        "exists": path.exists(),
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
        "num_rows": text.count("\\\\"),
        "num_columns": text.count("&"),
        "readable_by_pandas": "NA",
        "has_required_columns": "\\begin{tabular}" in text or "\\toprule" in text,
        "is_empty": len(text.strip()) == 0,
        "source_generation_script": "scripts/plot_final_grain_can.py or revision scripts",
    }


def audit_md(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    return {
        "file_type": "md",
        "file_path": str(path),
        "exists": path.exists(),
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
        "num_rows": text.count("\n"),
        "num_columns": "NA",
        "readable_by_pandas": "NA",
        "has_required_columns": True,
        "is_empty": len(text.strip()) == 0,
        "source_generation_script": "manual/generated report",
    }


def audit_svg(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    expect_legend = path.name in EXPECTED_LEGEND
    has_legend = any(word in text for word in ["legend", "ctt_test", "Recall", "F1"])
    return {
        "file_type": "svg",
        "file_path": str(path),
        "exists": path.exists(),
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
        "num_rows": "NA",
        "num_columns": "NA",
        "readable_by_pandas": "NA",
        "has_required_columns": "NA",
        "is_empty": len(text.strip()) == 0,
        "contains_svg_tag": "<svg" in text,
        "has_axes_text": any(token in text for token in ["F1", "Recall", "Feature", "Event", "Score", "delta_t", "payload", "pass", "risk", "Count"]),
        "has_legend_text_if_expected": (has_legend if expect_legend else True),
        "source_generation_script": "scripts/plot_final_grain_can.py or revision scripts",
    }


def main() -> None:
    AUDITS.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(TABLES.glob("*.csv")):
        rows.append(audit_csv(path))
    for path in sorted(TABLES.glob("*.tex")):
        rows.append(audit_tex(path))
    for path in sorted(FIGS.glob("*.svg")):
        rows.append(audit_svg(path))
    for path in sorted(ROOT.glob("*.md")) + sorted(AUDITS.glob("*.md")):
        rows.append(audit_md(path))
    out = pd.DataFrame(rows)
    out.to_csv(AUDITS / "output_file_integrity_audit.csv", index=False)
    bad = out[
        (out["exists"] == False)
        | (out["file_size_bytes"].fillna(0).astype(int) == 0)
        | (out["is_empty"] == True)
        | (out.get("readable_by_pandas", pd.Series(index=out.index, dtype=object)) == False)
        | (out.get("has_required_columns", pd.Series(index=out.index, dtype=object)) == False)
        | (out.get("contains_svg_tag", pd.Series(True, index=out.index)) == False)
        | (out.get("has_axes_text", pd.Series(True, index=out.index)) == False)
        | (out.get("has_legend_text_if_expected", pd.Series(True, index=out.index)) == False)
    ].copy()
    (AUDITS / "output_file_integrity_audit.md").write_text(
        "# Output File Integrity Audit\n\n"
        f"Checked files: {len(out)}\n\n"
        f"Problem files: {len(bad)}\n\n"
        + ("```csv\n" + bad.to_csv(index=False) + "```\n" if len(bad) else "No blocking integrity failures detected.\n"),
        encoding="utf-8",
    )
    print(f"[audit_final_grain_outputs] checked={len(out)} problems={len(bad)}")
    raise SystemExit(1 if len(bad) else 0)


if __name__ == "__main__":
    main()
