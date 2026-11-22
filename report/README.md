# report/

| Path | Content |
|---|---|
| `01_reporte/main.pdf` | Final report (IEEE conference template, English). Source: `main.tex`, `refs.bib`, `figures/`. |
| `02_presentacion/slides.pdf` | 10-minute presentation (Beamer, 16:9). Source: `slides.tex`. |
| `03_resumen/resumen.pdf` | One-page summary. Source: `resumen.tex`. |
| `contexto_tesis_Alberto.md` | Alberto's thesis context and next step (Spanish and English). |
| `Final project protocol ... .docx` | Protocol and status document. Its numbers predate the final runs; the PDFs above are the reference. |

No number in the documents is typed by hand. They come from `results/` through `01_reporte/numeros.tex` and the `tabla_*.tex` tables, which `generar_numeros.py` writes. The slides and summary reuse the report's macros and figures (`../01_reporte/`).

To regenerate after any change in the results:

```bash
python evaluate.py
python src/figuras_reporte.py
cd report/01_reporte && python generar_numeros.py && latexmk -pdf main.tex
cd ../02_presentacion && latexmk -pdf slides.tex
cd ../03_resumen && latexmk -pdf resumen.tex
```
