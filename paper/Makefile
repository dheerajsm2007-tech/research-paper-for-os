# Build the IEEE conference paper: pdflatex -> bibtex -> pdflatex -> pdflatex.
# Run from this directory: `make`. Regenerate the tables first with `make tables`.

PDFLATEX = pdflatex -interaction=nonstopmode -halt-on-error
MAIN     = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex refs.bib $(wildcard tables/*.tex)
	$(PDFLATEX) $(MAIN)
	bibtex $(MAIN)
	$(PDFLATEX) $(MAIN)
	$(PDFLATEX) $(MAIN)

tables:
	cd .. && python src/make_tables.py

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).pdf

.PHONY: all tables clean
