# Format fix report

- Preserved LNCS-style `\keywords{... \and ...}`.
- Added safer PDF text extraction support with `glyphtounicode`, `\pdfgentounicode=1`, UTF-8 input, T1/text companion encoding, and Latin Modern fonts.
- Changed acknowledgements and disclosure headings to starred headings inside `credits`, avoiding numbered 9.0.1/9.0.2 sections.
- Shortened figure captions and kept detailed protocol caveats in body text.
- Replaced over-wide result displays with resized or compact tables.
- Kept bibliography in one `thebibliography` block and added recent related-work entries consistently.
