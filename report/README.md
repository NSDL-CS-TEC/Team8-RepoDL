# report/

| Path | Content |
|---|---|
| `01_reporte/main.pdf` | Final report (IEEE conference template, English). |
| `02_presentacion/slides.pdf` | 10-minute presentation (Beamer, 16:9). |
| `03_resumen/resumen.pdf` | One-page summary. |
| `contexto_tesis_Alberto.md` | Alberto's thesis context and next step (Spanish and English). |
| `archive/protocol_2026-11-16.docx` | Protocol and status document as of 16 Nov 2026, kept as history. Its numbers predate the final runs; the PDFs above are the reference. |

The LaTeX sources (`*.tex`, `refs.bib`) are not versioned; the authors keep them locally. No number in the documents is typed by hand: `01_reporte/generar_numeros.py` writes them from `results/` into `numeros.tex` and the `tabla_*.tex` tables, which the sources include.

With the sources in place, regenerate after any change in the results:

```bash
python evaluate.py
python src/figuras_reporte.py
cd report/01_reporte && python generar_numeros.py && latexmk -pdf main.tex
cd ../02_presentacion && latexmk -pdf slides.tex
cd ../03_resumen && latexmk -pdf resumen.tex
```
