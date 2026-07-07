# GRAIN-CAN v22 Revision Report

## Changes

1. Restored the reference section to the LNCS/default bibliography style.
   - Removed the forced `\fontsize{5.5}{5.75}\selectfont` setting.
   - Removed aggressive negative bibliography spacing.
   - References are now rendered at normal readable size instead of compressed/overlapped text.

2. Relaxed several forced `[H]` figure placements in the evaluation section.
   - Converted evaluation figures to normal top floats where appropriate.
   - This allows LaTeX to place figures more naturally and reduces blank-space waste caused by forced float placement.

3. Recompiled and visually checked the PDF renders.
   - The reference section now spans pages 16--17 and is readable.
   - No undefined citation/reference warnings were introduced.

## Notes

- The paper is now 17 pages because the bibliography is no longer artificially compressed.
- This follows the LNCS-style bibliography behavior more closely than the previous compressed version.
