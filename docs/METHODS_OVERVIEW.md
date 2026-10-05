# Methods overview

ReproFly-AI evaluates an LLM-based autonomous scientist under blinding on a
real connectomics substrate. This document summarises the method; full details
are in the manuscript Methods and the frozen specification documents of the
release.

## Substrate and blinding

The complete male *Drosophila* CNS connectome is frozen as a canonical graph
of 11,657 anonymous cell types with three sealed ground-truth sets: PCD
(13 members), LC10a (1), CRZ (1). The Scientist agent (GLM-5.3) interacts
exclusively through a tool-mediated surface — candidate listing, local
inspection of top-10 neighbours, virtual silencing with an oracle-computed
outcome, experiment history, and final submission. It receives no ground-truth
information, no real biological type names, and no full node/edge dump. The
frozen system prompt and subagent configuration ship in `config/agents/`.

## Oracle

A structural-transmission oracle (first-passage and mass-balance machinery
over the type graph) computes, for any silencing event, a deterministic effect
estimate on pre-registered reproductive endpoint proxies. It is a structural
proxy, not physiology. Implementation: `src/reprofly/oracle/`.

## Scoring

Discovery is scored post hoc as AUTSRC — the area under the target-set
recovery curve on the 300-charged-attempt axis — by a sealed scorer
(`src/reprofly/benchmark/`), scoring implementation v1.1-corrected with a
recorded correction manifest. A size-adjusted first-hit random null
(10,000 permutations, master seed 20260925) provides the reference
distribution.

## Study phases

- **Phase 2A (formal benchmark):** 10 independent GLM-5.3 runs x 300 charged
  attempts against random and full-information baselines.
- **Phase 2B (post-hoc, exploratory):** 13 pre-specified behaviour metrics
  (M1-M13) over the frozen run bundles; every p-value labelled exploratory.
- **Phase 2C.1 (post-hoc matched-information controls):** seven non-LLM arms
  (C1 random walk, C2 greedy local, C3 frontier greedy, D1 record-minimum
  restart distilled from the observed LLM behaviour, D2a/D2b response-gated
  restarts, D3 map-then-mine), 100 runs x 300 attempts each, on the identical
  blinded surface and oracle.

## Provenance

Every number in the paper resolves to a single machine-readable layer
(`paper/results/PAPER_NUMBERS.json`), derived by a hash-recorded script from
authoritative frozen artifacts. Superseded and aborted artifacts are excluded
from all results and retained as provenance in the internal audit chain.
