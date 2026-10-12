"""Shared helpers for external-dataset study layers.

`studies/ops/` prepares the atlas this analysis was developed on. This package
prepares the public datasets used to test whether the criterion transfers, and it
sits beside `studies/ops/` rather than inside it because nothing here may depend
on OPS-specific structure.

Two rules are enforced here rather than left to each script.

**Locations are never guessed.** External data is not redistributed with this
repository. A script that inferred a path would turn a missing dependency into a
confusing read error somewhere further down, so every location arrives through an
argument or a named environment variable and is reported when absent.

**Feature attribution is an allowlist.** Deciding which columns of a morphological
profile may serve as the cheap input is the single place where the expensive
measurement can leak in, and channel dependence hides in more forms than a
denylist can enumerate: a cross-channel ``Correlation_RWC_Brightfield_Mito``, a
lowercase ``Image_Threshold_FinalThreshold_mito_bw``, an ``OrigAGP`` quality
metric, and every ``AreaShape`` feature whose mask was drawn on a fluorescence
channel. :func:`attribute_channels` therefore admits a column only when it is
positively attributable to the declared cheap channels, and everything else is
excluded and counted.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

__all__ = [
    "CellProfilerChannels",
    "attribute_channels",
    "partition_features",
    "resolve_source",
    "reject_repo_paths",
]


def resolve_source(
    cli_value: str | os.PathLike[str] | None,
    *,
    what: str,
    env_var: str,
    flag: str,
    default: str | None = None,
) -> str:
    """Resolve an external location from an argument, an environment variable or a declared default.

    Args:
        cli_value: the script's command-line value, if given.
        what: short description used in the error message.
        env_var: environment variable consulted when ``cli_value`` is absent.
        flag: argument name quoted back to the caller.
        default: a public, citable default such as an anonymous S3 endpoint. A
            local filesystem path is never a valid default.

    Raises:
        SystemExit: when nothing supplies a value, naming both the flag and the
            environment variable.
    """
    value = cli_value or os.environ.get(env_var) or default
    if not value:
        raise SystemExit(
            f"Could not locate {what}. Pass {flag} or set {env_var}. This location is not "
            "redistributed with this repository; see studies/external/README.md."
        )
    return str(value)


def reject_repo_paths(out: str | os.PathLike[str], *, flag: str = "--output-dir") -> Path:
    """Refuse an output directory inside the repository.

    Prepared tables are large, dataset-specific and untracked. Writing them into
    the checkout would put data in the source tree, which the release gate blocks
    and which makes a working copy impossible to reason about.
    """
    resolved = Path(out).expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    if resolved == repo or repo in resolved.parents:
        raise SystemExit(
            f"{flag} may not point inside the repository ({repo}): {resolved}\n"
            "External data and prepared tables are never committed. Write to a location "
            "outside the checkout."
        )
    return resolved


class CellProfilerChannels:
    """Channel vocabulary for one CellProfiler feature table.

    ``names`` are the channel tokens as they appear in column names, for example
    ``("Brightfield", "DNA", "Mito", "AGP", "RNA", "ER")``. Matching is
    case-insensitive and accepts the ``Orig`` and ``Corr`` prefixes CellProfiler
    emits, but requires a token boundary so that ``ER`` does not match inside
    ``Center`` or ``Number``.
    """

    def __init__(self, names: tuple[str, ...]) -> None:
        if not names:
            raise ValueError("at least one channel name is required")
        self.names = tuple(names)
        alternation = "|".join(sorted(map(re.escape, names), key=len, reverse=True))
        self._token = re.compile(rf"(?:orig|corr)?({alternation})", re.IGNORECASE)
        self._canonical = {name.casefold(): name for name in names}

    def of(self, column: str) -> frozenset[str]:
        """Channels this column is attributable to, matching on token boundaries."""
        found = set()
        for part in re.split(r"[^A-Za-z0-9]+", column):
            match = self._token.fullmatch(part)
            if match:
                found.add(self._canonical[match.group(1).casefold()])
        return frozenset(found)


def attribute_channels(columns: list[str], channels: CellProfilerChannels) -> dict[str, frozenset[str]]:
    """Map every column to the set of channels it is attributable to."""
    return {column: channels.of(column) for column in columns}


def partition_features(
    columns: list[str],
    *,
    channels: CellProfilerChannels,
    input_channels: tuple[str, ...],
    target_channels: tuple[str, ...] = (),
) -> dict[str, list[str]]:
    """Split feature columns into input, target and excluded, by positive attribution.

    A column joins ``input`` only when its channel set is non-empty and lies
    entirely inside ``input_channels``. A column joins ``target`` under the same
    rule against ``target_channels``. Everything else is excluded, including:

    * cross-channel columns that mention both an input and a target channel;
    * columns attributable to no channel at all, which in a Cell Painting table
      are segmentation-derived (``AreaShape``, ``Threshold``, object counts) and
      therefore depend on masks drawn on the expensive channels.

    The channel-free exclusion is the conservative choice and it is deliberate.
    Those features are not obtainable from the cheap channel alone in practice, so
    admitting them would overstate what a label-free measurement can do.

    Returns:
        A dict with keys ``input``, ``target``, ``excluded_cross_channel``,
        ``excluded_other_channel`` and ``excluded_channel_free``. The five lists
        partition ``columns`` exactly.
    """
    overlap = set(input_channels) & set(target_channels)
    if overlap:
        raise ValueError(
            f"input and target channels overlap, so the target would leak into the input: {sorted(overlap)}"
        )
    unknown = (set(input_channels) | set(target_channels)).difference(channels.names)
    if unknown:
        raise ValueError(f"channels not in the declared vocabulary: {sorted(unknown)}")

    inputs, targets = set(input_channels), set(target_channels)
    result: dict[str, list[str]] = {
        "input": [],
        "target": [],
        "excluded_cross_channel": [],
        "excluded_other_channel": [],
        "excluded_channel_free": [],
    }
    for column in columns:
        found = channels.of(column)
        if not found:
            result["excluded_channel_free"].append(column)
        elif found <= inputs:
            result["input"].append(column)
        elif targets and found <= targets:
            result["target"].append(column)
        elif found & inputs and found - inputs:
            result["excluded_cross_channel"].append(column)
        else:
            result["excluded_other_channel"].append(column)

    total = sum(len(value) for value in result.values())
    if total != len(columns):
        raise AssertionError(f"partition lost columns: {total} of {len(columns)}")
    if not result["input"]:
        raise ValueError(
            "no column is positively attributable to the declared input channels "
            f"{sorted(inputs)}; the channel vocabulary is probably wrong for this table"
        )
    return result


def frame_feature_columns(frame: pd.DataFrame, *, metadata_prefix: str = "Metadata_") -> list[str]:
    """Feature columns of a CellProfiler profile, that is, everything not metadata."""
    return [column for column in frame.columns if not column.startswith(metadata_prefix)]
