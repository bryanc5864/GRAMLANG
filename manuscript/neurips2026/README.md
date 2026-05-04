# NeurIPS 2026 Submission

Main track, double-blind, anonymized.

`main.pdf` is **not committed** (kept locally only; uploaded to OpenReview separately). Rebuild from source with `pdflatex` + `bibtex` + 2x `pdflatex`.

## Build

```
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Produces `main.pdf` (~14 pages: 9-page main body + references + appendix + checklist).

## Files

- `main.tex` — paper source
- `refs.bib` — references
- `checklist.tex` — NeurIPS Paper Checklist (filled)
- `neurips_2026.sty` — official NeurIPS 2026 style file (do not modify)
- `neurips_2026.tex` — upstream template (reference only)
- `figures/` — six manuscript figures, copied from `results/manuscript_figures/`

## Pre-submission checklist

- [x] All 16 NeurIPS Paper Checklist questions answered with justifications
- [x] Type 1 / embedded TrueType fonts only (no Type 3)
- [x] US Letter page size, line numbers, double-blind anonymized
- [x] Main body within 9-page limit
- [ ] All co-author OpenReview profiles created (moderation up to 2 weeks)
- [ ] Anonymized code mirror at `anonymous.4open.science`; replace footnote URL in `main.tex`
- [ ] Final read-through for any de-anonymizing references
- [ ] Submit title + abstract by **May 4, 2026 AOE**
- [ ] Submit full paper + supplementary by **May 6, 2026 AOE**

## Submission package

To assemble the OpenReview submission:

1. Build `main.pdf` locally.
2. Zip the supplementary: `figures/`, `main.tex`, `refs.bib`, `checklist.tex`, `neurips_2026.sty`, plus an anonymized snapshot of the code (no author info, no real GitHub URLs).
3. Upload `main.pdf` and the supplementary ZIP to OpenReview.
