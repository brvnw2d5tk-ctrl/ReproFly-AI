# ReproFly-AI

ReproFly-AI evaluates blinded, tool-mediated target search in the male
*Drosophila* CNS connectome. The evaluated agent was GLM-5.3. This public
package contains the benchmark library, frozen paper-level results, figure
sources and rendering code for offline reproduction.

## Reproduce the reported results

Python 3.11+ is required. Install the pinned environment from
`requirements-lock.txt` or `environment.yml`, then run:

```bash
python scripts/reproduce_results.py
python scripts/reproduce_figures.py
python scripts/reproduce_tables.py
python scripts/verify_release.py
```

The four scripts use bundled paper-level data and make no LLM API calls.
Rendered figures appear in `output/figures/`. The checked-in figure PDFs are
in `paper/figures/`; per-panel data are in `paper/source_data/`.

## Contents and scope

- `src/reprofly/`: benchmark, oracle, blinding and scoring library.
- `figure_v2/`: source-driven renderer for Figures 1–5.
- `paper/`: frozen results, figure data and figures.
- `scripts/`: offline reproduction scripts and supplementary reconstruction wrappers.
- `config/agents/`: studied agent's documented prompt and configuration.
- `manifests/SHA256SUMS.tsv`: file-integrity manifest.

The private-state test suite and member-level ground-truth mappings are not
included. The offline paper-level reproduction above uses the bundled frozen
data. The supplementary Level-2 wrappers are not a complete, independently
runnable reconstruction: source-data identifiers and checksums are not yet
final, and member-level ground truth remains controlled. See the manuscript
for the access conditions and AI-use disclosure.

See `CITATION.cff` for citation metadata and `LICENSE` for terms.
