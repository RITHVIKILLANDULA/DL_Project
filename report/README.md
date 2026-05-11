# Report — NeurIPS 2026 LaTeX

Structure matches the NeurIPS 2026 template the course provided: a main `.tex` paper that `\input`s a separate `checklist.tex` file, alongside the `.sty` and `.bib`.

## Files in this folder

| File | Purpose |
|---|---|
| `neurips_2026.tex` | Main paper source (sections 1–7, appendices, calls `\input{checklist}` at the end) |
| `checklist.tex` | NeurIPS Paper Checklist (16 items, all answered) — included into the main paper via `\input` |
| `neurips_2026.sty` | NeurIPS style file (formatting, page geometry, `\answerYes` macros) |
| `references.bib` | BibTeX bibliography (11 entries) |
| `fig_metrics_with_ci.png` | Figure: F1 with bootstrap 95% CIs |
| `fig_per_dataset.png` | Figure: accuracy by source dataset |
| `fig_length_bins.png` | Figure: accuracy by text-length bin |
| `demo_screenshots/` | 5 screenshots of the running Flask demo |

## Compile

```bash
cd report
pdflatex neurips_2026.tex
bibtex neurips_2026
pdflatex neurips_2026.tex
pdflatex neurips_2026.tex
```

Output: `neurips_2026.pdf`.

Or use Overleaf (zero install):
1. Upload all files in this folder as a new Overleaf project.
2. Set `neurips_2026.tex` as the main document (Menu → Main document).
3. Click **Recompile**. Download the PDF.

## What counts toward the 5–7 page limit

Per the instructor clarification: **only the main body** (sections 1–7, ending with Conclusion) counts. The following are excluded:

- References
- Appendix A (Compute Budget and Reproducibility)
- Appendix B (Demo)
- Team Contributions
- NeurIPS Paper Checklist (`checklist.tex`)

Our main body currently runs at exactly 5 pages — comfortably within the 5–7 limit.

## What's pre-filled

- All numbers (Tables 3–5, all figures) taken **verbatim** from `reports/full_evaluation.json`, `reports/multiseed_aggregate.json`, and `reports/error_analysis_detailed.json`.
- Authors: all three team members listed with emails.
- GitHub link: https://github.com/RITHVIKILLANDULA/DL_Project
- Team Contributions section, NeurIPS Paper Checklist (all 16 items) — all filled in.
