# NeurIPS 2026 Submission

Main track, double-blind. Anonymized.

## Build

```
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Outputs `main.pdf`.

## Files

- `main.tex` — paper source (~9 pages main body + appendix + checklist)
- `refs.bib` — references
- `checklist.tex` — NeurIPS Paper Checklist (filled, do not remove)
- `neurips_2026.sty` — official NeurIPS 2026 style file (do not modify)
- `neurips_2026.tex` — original template (kept for reference; not the submission)
- `figures/` — six manuscript figures, copied from `results/manuscript_figures/`

## Pre-submission checklist

- [ ] Verify all numerical claims against `results/` JSONs
- [ ] Ensure all author OpenReview profiles exist (moderation up to 2 weeks)
- [ ] Set up `anonymous.4open.science` mirror; replace footnote URL in `main.tex`
- [ ] Final read-through for any de-anonymizing references
- [ ] Submit title + abstract by **May 4, 2026 AOE**
- [ ] Submit full paper + supplementary by **May 6, 2026 AOE**
