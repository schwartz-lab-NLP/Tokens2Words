# Build and audit targets for the technical note.
# `make pdf`   - compile with latexmk (same toolchain Overleaf uses)
# `make count` - body word count (tables/bib excluded via %TC:ignore fences)
# `make audit` - checks no unfilled \PH placeholders remain (results are filled)
# `make zip`   - Overleaf-ready archive with main.tex at the zip root
# `make clean` - remove build products

.PHONY: pdf count audit zip clean

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

count:
	texcount -brief -sum main.tex

audit:
	@if grep -nE '\\PH\{' main.tex; then echo "FAIL: unfilled placeholders above"; exit 1; else echo "PASS: no unfilled placeholders"; fi

zip: pdf
	zip -r tokenizer_mismatch_note_overleaf.zip main.tex references.bib \
		iclr2026_conference.sty iclr2026_conference.bst figures/ -x '*.DS_Store'

clean:
	latexmk -C
	rm -f tokenizer_mismatch_note_overleaf.zip

SHELL := /bin/bash
