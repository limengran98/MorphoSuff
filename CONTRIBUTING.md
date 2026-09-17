# Contributing

Contributions to the reusable package, dataset adapters and study workflows are
welcome. Please open an issue before a large change to agree on its scope.

## Development setup

Install the development dependencies and run the available checks. Tests that
require unavailable optional dependencies are marked and skip with a reason.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest -ra
```

For changes to OPS runners, neural models or raw-phase entry points, also run
the suite with the model and OPS data dependencies installed.
`MEASUREMENT_SUFFICIENCY_REQUIRE_OPTIONAL=1` requires the marked optional
dependencies to be available.

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -e ".[test,models,ops-data]"
MEASUREMENT_SUFFICIENCY_REQUIRE_OPTIONAL=1 python -m pytest -ra
```

These are local checks for package behaviour and scientific data contracts.
Manuscript wording, Supplementary Information numbering and figure layout are
not part of the test suite. Figure build entry points are documented in the
[figure-source guide](paper/figure_sources/README.md).

## Pull-request requirements

- Add a focused test, including a known-answer test for any scientific
  aggregation or threshold change.
- Fit preprocessors only on training partitions and add a leakage test for a
  new split.
- Keep sparse target availability as an explicit mask; never encode an absent
  assay as zero.
- State the statistical unit, exclusions, missing mappings and uncertainty
  unit for a new analysis.
- Preserve model identity through validation. Do not treat model variants as
  biological replicates.
- Keep large training datasets, prediction trees and model weights outside
  version control. The release under `paper/` includes compact source tables
  and plotting code; preserve their documented provenance when updating them.
  Exclude signed URLs, credentials and
  machine-specific filesystem paths from all contributions.
- Mark tests that need optional dependencies with `requires_torch`,
  `requires_catboost` or `requires_scipy`. Apply a marker only when the test
  uses that dependency; standard-library provenance and contract checks can
  run independently.

## Adding a dataset adapter

Implement `MeasurementDatasetAdapter`, declare `AdapterCapabilities`, emit the
canonical tables and add foreign-key/cardinality tests. Check the required
capabilities before running each analysis. Keep dataset identifiers and
calibrated measurement thresholds specific to the new resource. See the
[dataset adaptation guide](docs/adapt_new_dataset.md).

## Adding an external model

An adapter must record the upstream repository, immutable revision, licence,
environment and exact adaptation boundary. It must accept the frozen split and
sampling manifests, provide an executable training or inference route, and
export the canonical long prediction table.

## Figure contributions

Figure code receives validated source tables rather than model output directories.
Every panel must document its statistical unit and source contract. Rendering
changes should not change scientific aggregation; aggregation changes require
new known-answer tests and a protocol revision.
