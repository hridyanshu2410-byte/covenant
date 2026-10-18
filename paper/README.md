# Paper sources

LaTeX sources of the journal paper (`journal/`) and its supplementary material
(`supplement/`), with the figures they include (`figures/`). The overview figure
is drawn from `figures/src/fig-overview.html`.

Build the journal paper first, since the supplement refers to its labels (TeX Live with `acmart` 2.16 or later; a copy of the class is included):

    cd journal    && pdflatex main && bibtex main && pdflatex main && pdflatex main
    cd supplement && pdflatex main && bibtex main && pdflatex main && pdflatex main

All numbers in the paper come from the released checker (see the top-level
README) and the scripts under `experiments/`.
