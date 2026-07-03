# Template Compliance Check

The user-provided `samplepaper.tex` is the reference. The final manuscript has two forms:

- `attack_centric_can_ids_paper.tex`: local build source with a fallback path because this machine does not provide `llncs.cls`.
- `results/final_submission_closure/template/attack_centric_can_ids_paper_lncs_submission.tex`: strict LNCS submission source generated from the same manuscript.

## Pass items

- Uses `\documentclass[runningheads]{llncs}` in the strict submission source.
- Uses `\usepackage[T1]{fontenc}` to avoid PDF text extraction and ligature issues.
- Uses `graphicx`.
- Uses LNCS title, author, running head, institute, and `\maketitle`.
- Keeps `\keywords{... \and ...}` inside the abstract.
- Uses the LNCS `credits` environment.
- Keeps captions above tables and below figures.
- Uses vector/PDF-compatible figures.
- The locally compiled revised PDF is 16 pages.

## Remaining compile note

The local environment does not contain `llncs.cls`, so the compiled PDF uses the local fallback source. For final submission, compile the strict source with the official CIVS/Springer template package. The content and layout decisions have already been aligned to the sample template.
