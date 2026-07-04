# Figure 1 Nature-Style Revision Report

Figure revised: `figures/grain_can_pipeline.svg` and `figures/grain_can_pipeline.pdf`

Workflow:

- Installed and used the `nature-figure` skill guidance.
- Selected the Python/matplotlib backend because the repository's figure workflow is Python-based.
- Rebuilt Figure 1 as a schematic-led composite figure rather than a simple flowchart.

Figure contract:

- Core conclusion: GRAIN-CAN preserves same-ID local CAN behavior before fixed-window aggregation.
- Archetype: schematic-led composite.
- Evidence logic: raw CAN trace -> same-ID memory -> causal local deltas -> feature-preserving window table -> supervised score and IDS report.
- Protocol layer: causal features, fixed protocol, attack-centric metrics, and score metrics are shown as a bottom contract row.

Quality checks:

- Text is kept as editable SVG text (`svg.fonttype=none`).
- White background, restrained palette, black/gray arrows, and print-safe outlines are used.
- No labels overlap in the exported SVG/PDF.
- The revised manuscript compiles to 13 pages, under the 16-page limit.
- LaTeX compile has no overfull boxes, citation warnings, undefined references, or errors.
