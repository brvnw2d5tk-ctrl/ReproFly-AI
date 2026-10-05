# Authoritative frozen artifacts (paper_release)

Byte-identical copies of the private authoritative result files, frozen here
so that third parties can verify every paper number without access to
derived_private. Excluded by design: raw connectome data, private GT/blind
mappings, provider logs, aborted attempts (provenance only).

Superseded / aborted artifacts retain their role labels in their filenames.

## Provenance note (sidecar — recorded in lieu of modifying sealed files)

- `phase2a_SEALED_SCORING_RESULTS_v1.1_CORRECTED.json` embeds
  `input_sha256["harness/sealed_scoring.py"]`. That recorded hash matches the
  ARCHIVED execution copy
  (`derived_private/formal_execution/harness/_archive_v1_0_execution/sealed_scoring.py.v1_0_as_executed`)
  byte-for-byte. The live harness file differs only by a 13-line comment block
  added after sealing; the recorded hash is the correct hash of the code that
  produced the sealed results.
- The scoring-correction manifest and integrity audit live in the internal
  repo (`derived_private/formal_execution/PHASE2C1_*`, `PHASE2A_SCORING_CORRECTION_MANIFEST.json`,
  `PHASE2A_SCORING_INTEGRITY_AUDIT.*`); their conclusions are reflected in
  `phase2a/PHASE2A_SUMMARY.json` (superseded_source block) without modifying
  any sealed artifact.
