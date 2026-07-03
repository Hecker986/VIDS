# Format fix report

- Preserved LNCS-style `\keywords{... \and ...}`.
- Added safer PDF text extraction support with `glyphtounicode`, `\pdfgentounicode=1`, UTF-8 input, T1/text companion encoding, and Latin Modern fonts.
- Kept the LNCS sample-paper structure: table captions above tables, figure captions below figures, `\keywords{... \and ...}` inside the abstract, and `\ackname`/`\discintname` inside `credits`.
- Added conservative float placement, compact float spacing, and top-aligned float pages to reduce large blank gaps without changing the research content.
- Shortened figure captions and kept detailed protocol caveats in body text.
- Replaced over-wide result displays with resized or compact tables.
- Kept bibliography in one `thebibliography` block and added recent related-work entries consistently.
