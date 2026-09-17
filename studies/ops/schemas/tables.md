# OPS canonical tables

The adapter emits five long-form tables.  Large arrays remain external and are
loaded lazily by identifiers in these tables.

| Table | Required columns | Invariant |
| --- | --- | --- |
| observations | `observation_id`, `input_cell_id`, `target_cell_id`, `assay_id`, `perturbation_id`, `guide_id`, `screen_id`, `well_id`, `field_id`, `is_control` | one row per assay-qualified paired observation; `assay_id` is an FK to `assays`; `guide_id` may be null |
| reporters | `reporter_id`, `reporter_name`, `biological_system`, `target_modality`, `endpoint_schema_id` | exactly 52 OPS reporters |
| assays | `assay_id`, `reporter_id`, `screen_id`, `endpoint_ids` | exactly 99 observed reporter-screen edges over 73 screens; endpoints are a non-empty reporter-specific list |
| pairing | `observation_id`, `input_cell_id`, `target_cell_id`, `assay_id`, `pairing_key`, `pairing_confidence` | exactly one linkage per observation; input and target cells are one-to-one within each assay |
| predictions | `observation_id`, `reporter_id`, `endpoint_id`, `split_name`, `fold`, `model_id`, `y_true`, `y_pred`, `screen_id`, `perturbation_id` | one declared prediction per model/fold/cell/endpoint |

`observation_id` identifies a reporter-assay observation, while
`input_cell_id` identifies the underlying phase cell. A phase cell may therefore
appear once in each of several assays when multiple reporters were measured on
that same cell. Global uniqueness of `input_cell_id` or `target_cell_id` would
discard this valid OPS topology. The frozen census makes this distinction
observable: it contains 7,344,374 independent phase cells but 9,996,286
assay-qualified reporter observations. Exact pairing is instead enforced as a
bijection within `assay_id`: `(assay_id, input_cell_id)`,
`(assay_id, target_cell_id)`, and `(assay_id, pairing_key)` are unique, and the
four linkage identifiers in `pairing` must exactly match `observations`.

The `assays` table carries reporter-specific endpoint masks and is the
authoritative route from an observation to its reporter and screen. The
duplicated `observations.screen_id` must equal the assay screen. Never
materialize missing reporter blocks as measured zeroes. Controls are matched
within screen when computing perturbation responses.
