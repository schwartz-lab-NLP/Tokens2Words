# The Price of a Mismatched Tokenizer: Distribution Shift as an Efficiency Tax

A short technical note, typeset in the ICLR 2026 conference style, quantifying the efficiency cost of
deploying a tokenizer trained on distribution *P* against end-user traffic
distribution *Q*. Experimental results are filled in from the completed companion
experiment (`../tokshift/results.json`, 2026-07-29); every other number is either
cited or derived (arithmetic in `derivations.md`). `placeholders_schema.txt` is
kept as the provenance map from the note's result cells to the experiment's
`results.json` keys.

## Files

| File | Purpose |
|---|---|
| `main.tex` | The complete note (preamble, §1–§5, tables, equations, figure slot) |
| `iclr2026_conference.sty` / `.bst` | Official ICLR 2026 style and BibTeX style (author–year citations) |
| `references.bib` | Bibliography; entries with unconfirmed fields carry `note={[verify]}` |
| `placeholders_schema.txt` | The 19 placeholder keys — must match the experiment's `results.json` exactly |
| `derivations.md` | Step-by-step arithmetic behind every "(derived)" number in the note |
| `figures/` | Slot for `fig1_inflation_vs_divergence.pdf` (see `figures/README.md`) |
| `Makefile` | `make pdf` / `count` / `audit` / `zip` / `clean` |

## Build locally

```bash
make pdf      # latexmk -pdf (TeX Live; same toolchain as Overleaf)
make count    # body word count, tables/bib excluded (target 1,100–1,300, hard cap 1,350)
make audit    # placeholder keys in main.tex must exactly equal placeholders_schema.txt
```

## Upload to Overleaf

```bash
make zip      # produces tokenizer_mismatch_note_overleaf.zip with main.tex at the root
```

On Overleaf: **New Project → Upload Project** → select the zip. Compiles out of the
box with pdfLaTeX (TeX Live 2024+); Overleaf runs the pdflatex → bibtex →
pdflatex ×2 cycle automatically. The ICLR style files ship inside the zip, so no
Overleaf-side template selection is needed.

The note uses `\iclrfinalcopy` so the author name is shown rather than the
double-blind placeholder, with the running head overridden to "Preprint" instead
of ICLR's "Published as a conference paper" banner. For an actual submission,
drop the `\lhead` override and comment out `\iclrfinalcopy`.

## Experimental results

All 19 result values are filled from `../tokshift/results.json` (rounded for
presentation; full precision in the JSON). Figure 1 is the experiment's
`fig1_inflation_vs_divergence.pdf`. Two provenance notes: the `JSD_NORM_*` schema
keys hold normalized byte-n-gram divergence D_norm (the name is a historical
misnomer), and `FERTILITY_DEPLOYED_Q`/`FERTILITY_RETRAINED_Q` are unweighted
means over the four Q corpora. The `\PH{KEY}` macro remains defined in the
preamble in case future revisions reintroduce placeholders.
