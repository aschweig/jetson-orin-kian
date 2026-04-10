#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode kian.tex
bibtex kian
pdflatex -interaction=nonstopmode kian.tex
pdflatex -interaction=nonstopmode kian.tex
