"""Portable measurement-sufficiency contracts, analyses, and falsification tools."""

from .adapters import AdapterCapabilities, EligibilityError, LocalManifestAdapter, MeasurementDatasetAdapter
from .consensus import FINAL_TEN_MODELS, consensus_median
from .low_label import LowLabelManifest, build_low_label_manifest
from .model_registry import MODEL_REGISTRY, ModelEligibilityError, create_model, create_ops_model, eligible_models
from .schemas import validate_canonical_tables, validate_predictions
from .sufficiency import TierThresholds, SufficiencyTier, assign_tier
from .training import TrainedSparseModel, fit_sparse_model, prediction_long_table

__all__ = [
    "AdapterCapabilities",
    "EligibilityError",
    "LocalManifestAdapter",
    "MeasurementDatasetAdapter",
    "FINAL_TEN_MODELS",
    "MODEL_REGISTRY",
    "ModelEligibilityError",
    "create_model",
    "create_ops_model",
    "eligible_models",
    "consensus_median",
    "LowLabelManifest",
    "build_low_label_manifest",
    "TrainedSparseModel",
    "fit_sparse_model",
    "prediction_long_table",
    "SufficiencyTier",
    "TierThresholds",
    "assign_tier",
    "validate_canonical_tables",
    "validate_predictions",
]

__version__ = "0.1.0"
