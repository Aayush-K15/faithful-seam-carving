# Paper

`paper.pdf` is the compiled paper (IEEE conference format). To rebuild:

```bash
python scripts/paper_figures.py      # from the repo root: regenerates figs/ (needs data/, see scripts/get_data.py)
cd paper && latexmk -pdf paper.tex
```

Or upload `paper.tex`, `refs.bib` and `figs/` to Overleaf (IEEEtran is built in).
Every number in Table I comes from `results/summary.csv`.
