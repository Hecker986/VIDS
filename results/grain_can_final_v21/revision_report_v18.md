# GRAIN-CAN v18 Revision Report

This revision focuses on clarifying detector names and baseline definitions.

## Changes

1. Added an explicit detector-name paragraph in the Compared Pipelines subsection.
   - `Sample-feature GB` is defined as public-style sample features plus Gradient Boosting.
   - `Sample-feature MLP` is defined as the same sample-level representation with a multilayer perceptron.
   - `Raw-window Transformer` is defined as the existing raw fixed-window neural baseline.
   - `Raw-window GB` is defined as raw W100 window statistics plus Gradient Boosting.
   - `GRAIN-CAN (GB)` is defined as same-ID residual W100 features plus the same Gradient Boosting classifier family.

2. Renamed code-like detector labels throughout the main text and tables.
   - `GB-sample` -> `Sample-feature GB`
   - `Raw-Trans` -> `Raw-window Transformer`
   - `Raw-GB` -> `Raw-window GB`
   - `GRAIN-GB` / `GRAIN-CAN+GB` -> `GRAIN-CAN (GB)`

3. Updated Table 3 caption and rows to explain that GB denotes Gradient Boosting and that the key controlled comparison is Raw-window GB versus GRAIN-CAN (GB).

4. Updated discussion text, result text, low-FPR analysis, and efficiency paragraphs to use the clarified detector names.

## Build status

The paper was recompiled successfully with no undefined citations/references and no overfull warnings reported in the compile log.
