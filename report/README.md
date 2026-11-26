# report/

| Path | Content |
|---|---|
| `01_reporte/main.pdf` | Final report (IEEE conference template, English). |
| `02_presentacion/slides.pdf` | 10-minute presentation (Beamer, 16:9). |
| `03_resumen/resumen.pdf` | One-page summary. |
| `contexto_tesis_Alberto.md` | Alberto's thesis context and next step (Spanish and English). |
| `archive/protocol_2026-11-16.docx` | Protocol and status document as of 16 Nov 2026, kept as history. Its numbers predate the final runs; the PDFs above are the reference. |

Only the PDFs are versioned. The LaTeX sources (`*.tex`, `refs.bib`), the script that writes their numbers from `results/` (`01_reporte/generar_numeros.py`) and their copy of the figures (`01_reporte/figures/`) are kept locally by the authors. The figures themselves are in `figures/` at the project root, and `python src/figuras_reporte.py` regenerates them.
