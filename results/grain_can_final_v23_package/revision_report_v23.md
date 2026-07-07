# v23 layout revision

This revision focuses only on layout and float/reference stability. It keeps the existing figures, body content, and bibliography entries.

Changes made:
- Kept all main figures, including Fig. 5 and Fig. 6.
- Reduced excessive float spacing and tuned LaTeX float placement (`topfraction`, `bottomfraction`, `textfraction`, and `floatpagefraction`).
- Slightly reduced the widths of late evaluation figures so they fit with surrounding text more naturally.
- Removed hard float barriers around the evaluation figures that could create underfilled pages, while keeping a barrier before Discussion so evaluation figures do not drift into Discussion.
- Re-rendered and checked pages after compilation.
- Fixed the bibliography layout by using a compact but non-overlapping bibliography setting, preventing the reference-list garbling seen in the earlier PDF and avoiding a nearly empty final references page.

Verification:
- Recompiled successfully with LaTeX.
- Rendered all pages to PNG for visual inspection.
- Confirmed that Fig. 5 and Fig. 6 are retained.
- Confirmed that the references are no longer visually overlapped.
- Confirmed that the final PDF has 16 pages.
