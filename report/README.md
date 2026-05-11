# Report — NeurIPS 2026 LaTeX

## Files in this folder

- `report.tex` — main report source
- `references.bib` — BibTeX citations
- `fig_metrics_with_ci.png` — Figure: F1 with bootstrap 95% CIs
- `fig_per_dataset.png` — Figure: accuracy by source dataset
- `fig_length_bins.png` — Figure: accuracy by text-length bin

## One file you need to add

Drop `neurips_2026.sty` (the style file you pasted, or download from https://neurips.cc/Conferences/2026 → Author Resources) into this folder.

## Compile

```bash
cd report
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

Output: `report.pdf`.

## What counts toward the 5–7 page limit

Per the instructor clarification: **only the main body** (sections 1–7, ending with Conclusion) counts. The following are excluded from the page limit:

- References
- Appendix A (Compute Budget and Reproducibility)
- Appendix B (Demo)
- LLM Usage Disclosure
- Team Contributions
- NeurIPS Paper Checklist

If the main body still runs over 7 pages after a first compile, the easiest trim is Section 3 (Method) — collapse the bulleted lists into prose paragraphs.

## What's pre-filled

- All numbers (Table 3, Table 4, Table 5, all figures) taken **verbatim** from `reports/full_evaluation.json`, `reports/multiseed_aggregate.json`, and `reports/error_analysis_detailed.json` — no rewriting.
- Authors: all three team members listed with emails.
- GitHub link: https://github.com/RITHVIKILLANDULA/DL_Project
- LLM Usage section, Team Contributions section, NeurIPS Paper Checklist (all 16 items) — all filled in.

## What's NOT in here

Nothing — the paper is complete as far as I can tell. Read through `report.tex` and let me know if any wording needs tweaking before you compile.
