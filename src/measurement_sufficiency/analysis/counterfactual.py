"""Interfaces for assay-deletion utility and external GO/complex overlap tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd


def validate_assay_deletion(*, deleted_assay_id: str, feature_provenance: pd.DataFrame, assay_column: str = "assay_id", feature_column: str = "feature_id") -> pd.DataFrame:
    """Fail closed if a deleted assay can re-enter through a proxy feature.

    The caller supplies a feature provenance table, so this works for arbitrary
    model implementations without pretending that a model's internals are
    inspectable.  Every input feature must have exactly one declared source.
    """
    needed = {assay_column, feature_column}
    if missing := needed.difference(feature_provenance):
        raise ValueError(f"feature provenance lacks columns: {', '.join(sorted(missing))}")
    if feature_provenance[feature_column].duplicated().any():
        raise ValueError("feature provenance must map each feature to exactly one source")
    provenance_ids = feature_provenance[assay_column]
    if provenance_ids.isna().any():
        raise ValueError(
            "feature provenance must declare a non-null source assay for every feature; "
            "a feature of unknown origin cannot pass a fail-closed deletion gate"
        )
    # Both sides are compared as strings. counterfactual_assay_deletion passes
    # str(assay_id) while this column may hold integers, and an .eq() across that
    # mismatch never fires, so the gate silently let a deleted assay back in.
    leaked = feature_provenance.loc[
        provenance_ids.astype(str).eq(str(deleted_assay_id)), feature_column
    ]
    if not leaked.empty:
        raise ValueError(f"deleted assay {deleted_assay_id!r} re-enters via features: {', '.join(map(str, leaked))}")
    return feature_provenance.copy()


def counterfactual_assay_deletion(assays: Iterable[str], *, feature_provenance: pd.DataFrame, evaluate: Callable[[str], pd.DataFrame], assay_column: str = "assay_id") -> pd.DataFrame:
    """Run a caller-owned deletion evaluator after strict proxy-leak checks.

    ``evaluate`` receives an assay id and must train/evaluate the deleted-assay
    protocol itself.  Returning a table keeps this module independent of the
    training layer and makes deletion provenance auditable.
    """
    results = []
    for assay_id in assays:
        validate_assay_deletion(deleted_assay_id=str(assay_id), feature_provenance=feature_provenance, assay_column=assay_column)
        result = evaluate(str(assay_id)).copy()
        result["deleted_assay_id"] = str(assay_id)
        results.append(result)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def go_complex_overlap(predicted_terms: pd.DataFrame, reference_terms: pd.DataFrame, *, entity_column: str = "gene_id", term_column: str = "term_id", group_columns: list[str] | None = None) -> pd.DataFrame:
    """Compute set overlap against externally supplied GO or complex annotations."""
    groups = group_columns or ["reporter_id"]
    needed = set(groups + [entity_column, term_column])
    for name, table in (("predicted_terms", predicted_terms), ("reference_terms", reference_terms)):
        if missing := needed.difference(table):
            raise ValueError(f"{name} lacks columns: {', '.join(sorted(missing))}")
    rows = []
    all_groups = pd.concat([predicted_terms[groups], reference_terms[groups]], ignore_index=True).drop_duplicates()
    for _, key in all_groups.iterrows():
        mask_pred = pd.Series(True, index=predicted_terms.index)
        mask_ref = pd.Series(True, index=reference_terms.index)
        for column in groups:
            mask_pred &= predicted_terms[column].eq(key[column])
            mask_ref &= reference_terms[column].eq(key[column])
        pred = set(map(tuple, predicted_terms.loc[mask_pred, [entity_column, term_column]].to_numpy()))
        ref = set(map(tuple, reference_terms.loc[mask_ref, [entity_column, term_column]].to_numpy()))
        union = pred | ref
        rows.append({**key.to_dict(), "predicted_n": len(pred), "reference_n": len(ref), "overlap_n": len(pred & ref), "jaccard": len(pred & ref) / len(union) if union else float("nan")})
    return pd.DataFrame(rows)
