# Paper sources

LaTeX sources of the journal paper (`journal/`) and its supplementary material
(`supplement/`), with the figures they include (`figures/`). Figures 1, 2, 7 and 8
are drawn as HTML in `figures/src/` and printed to vector PDF with Chromium
(`render.js`, Playwright; fonts Libertinus and Inconsolata):

    cd figures/src && python3 build_teaser.py && python3 build_precision.py && python3 build_repairs.py
    NODE_PATH=<node_modules with playwright> node render.js fig1-teaser fig-overview fig-precision fig-repairs

The numbers in them are read from `repairs.json` (per-cause diagnostics of the
repair cases, `experiments/repairs`), `precision.json` (classes of the fresh
studies, `experiments/fresh-studies/data`) and `teaser-evidence/` (the SDL3
trace and verbatim checker output). The frames of the programs (`img/`) are
lavapipe screenshots of the same commands: `cap.py` runs one program under
Xvfb and grabs its window, `capall.sh` and `caplist.py` list the repair and
study programs, and `prep_img.py` crops the tiles.

Build the journal paper first, since the supplement refers to its labels (TeX Live with `acmart` 2.16 or later; a copy of the class is included):

    cd journal    && pdflatex main && bibtex main && pdflatex main && pdflatex main
    cd supplement && pdflatex main && bibtex main && pdflatex main && pdflatex main

All numbers in the paper come from the released checker (see the top-level
README) and the scripts under `experiments/`.
