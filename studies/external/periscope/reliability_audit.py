#!/usr/bin/env python3
"""Audit the PERISCOPE reliability estimate: its pooling rule, then its gene set.

``reliability`` is the gate that decides both PERISCOPE verdicts, and it decides them
at the first test, before any of the four performance quantities is consulted. A
number carrying that much weight has to be checked against the two things that could
make it an artefact rather than a measurement.

**One: the pooling rule.** ``response_magnitudes`` standardises each endpoint before
taking a norm, and ``_ridge.recoverability_by_fold`` was changed to match on
2026-08-14, because pooling hundreds of endpoints in native CellProfiler units
describes whichever endpoint has the largest units rather than the block. The
reliability path was left in native units, so the rule was comparing a value and its
ceiling formed under different rules. ``_reliability`` now applies one rule to both;
this script reports the estimate under each so the change is checkable rather than
asserted.

**Two: the gene set.** Reliability is a split-half correlation over (gene, endpoint)
pairs, so it is ``Var(true) / (Var(true) + Var(noise))`` on whatever gene set it is
computed over. PERISCOPE is genome scale at 20,389 genes, the large majority of which
have no effect on any mitochondrial measurement, so both halves are mostly noise
around zero. The atlas that calibrated the 0.30 floor carries 1,000 genes, which a
targeted library selects rather than samples. If the difference matters, the floor and
the estimate are not comparable and the verdict is a property of library design.

Restricting to a gene set is evidence only if the restriction is declared in advance
and chosen without looking at the target values, and only if it is separated from the
mechanical effect of using fewer genes. Both are enforced: gene sets arrive as files
from public releases, and every gene set is accompanied by ``--n-random`` size-matched
random subsets drawn from the same screen. Without that control a gene-set result
cannot be read at all, which is why ``--n-random 0`` is refused.

Nothing here refits a model. The split-half blocks are built once and every gene set
is a row selection on them, so the random controls cost almost nothing.

Gene-set files are not redistributed with this repository. Each is a text file whose
first tab-separated field is an HGNC symbol, with an optional header line, and each is
recorded in the output by sha256. The two used in the manuscript:

    CEGv2       https://raw.githubusercontent.com/hart-lab/bagel/master/CEGv2.txt
                Hart et al., core essential genes, 684 symbols. Target agnostic, so it
                applies unchanged to every reporter, and it is the closest public
                analogue of a library chosen because its members do something.
    MitoCarta3  https://personal.broadinstitute.org/scalvo/MitoCarta3.0/Human.MitoCarta3.0.xls
                sheet "A Human MitoCarta3.0", column Symbol, 1,136 symbols. Specific to
                this reporter, which is an anti-TOMM20 channel.

Example:

    python studies/external/periscope/reliability_audit.py \\
      --prepared ../external/periscope_guide \\
      --output-dir ../external/periscope_reliability_audit \\
      --gene-set CEGv2=../genesets/cegv2.symbols.txt \\
      --gene-set MitoCarta3=../genesets/mitocarta3.symbols.txt \\
      --n-random 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from _common import reject_repo_paths  # noqa: E402

from _reliability import (  # noqa: E402
    reliability_by_block,
    split_half_blocks,
    summarise_reliability,
)

from measurement_sufficiency.analysis.tiers import OPS_TIER_RULES  # noqa: E402

REPORTER_ID = "tomm20_mito"

#: Headers a gene-set file may carry in its first field, matched case-insensitively.
GENE_SET_HEADERS = {"gene", "genes", "symbol", "symbols", "hgnc_symbol", "gene_symbol"}

_KEYS = ["screen_id", "perturbation_id", "guide_id", "is_control"]


def load_targets(prepared: Path) -> tuple[pd.DataFrame, list[str], str]:
    """Guide-level target rows, whichever level the prepared directory holds.

    The input block is never read. ``analyse_guide.load`` merges the 2,508 input
    features and ``analyse_cells.load_cells`` reads 15 GB, neither of which reliability
    uses.

    A single-cell directory is collapsed here exactly as ``analyse_cells`` collapses it
    before estimating reliability, and in the same row order, so the number this audit
    reports under native pooling can be checked against the number that script already
    wrote. Both levels then present the same shape and the rest of the audit does not
    know which it has.
    """
    manifest = json.loads((prepared / "canonical_manifest.json").read_text(encoding="utf-8"))
    level = manifest.get("aggregation_level")
    endpoints = list(manifest["target_endpoint_columns"])
    if level == "guide":
        observations = pd.read_parquet(
            prepared / manifest["tables"]["observations"], columns=["observation_id", *_KEYS]
        )
        targets = pd.read_parquet(prepared / manifest["tables"]["targets"])
        return observations.merge(targets, on="observation_id", how="inner"), endpoints, level
    if level != "single_cell":
        raise SystemExit(f"unsupported aggregation_level {level!r}")

    observations = (
        pd.read_parquet(prepared / manifest["tables"]["observations"],
                        columns=["observation_id", *_KEYS])
        .sort_values(["screen_id", "observation_id"], kind="stable")
        .reset_index(drop=True)
    )
    root = prepared / manifest["tables"]["targets"]
    blocks = []
    for screen, block in observations.groupby("screen_id", sort=True):
        files = sorted(root.glob(f"plate={screen}.*.parquet"))
        if not files:
            raise SystemExit(f"screen {screen} has no target partition on disk")
        rows = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        rows = rows.set_index("observation_id").reindex(block["observation_id"])
        if rows[endpoints[0]].isna().all():
            raise SystemExit(f"screen {screen}: observation identifiers do not match the partition")
        frame = pd.concat(
            [block[_KEYS].reset_index(drop=True), rows[endpoints].reset_index(drop=True)], axis=1
        )
        collapsed = frame.groupby(_KEYS, as_index=False, sort=True)[endpoints].mean()
        collapsed[endpoints] = collapsed[endpoints].astype(np.float32)
        blocks.append(collapsed)
        print(f"  collapsed {screen}: {len(block):,} cells to {len(collapsed):,} guide profiles",
              flush=True)
        del rows, frame
    return pd.concat(blocks, ignore_index=True), endpoints, level


def read_gene_set(path: Path) -> tuple[set[str], str]:
    """Symbols from a one-per-line or tab-separated file, and the file's sha256."""
    raw = path.read_bytes()
    lines = [line.strip() for line in raw.decode("utf-8").splitlines() if line.strip()]
    symbols = [line.split("\t")[0].strip() for line in lines]
    if symbols and symbols[0].casefold() in GENE_SET_HEADERS:
        symbols = symbols[1:]
    if not symbols:
        raise SystemExit(f"{path} carries no gene symbols")
    return set(symbols), hashlib.sha256(raw).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output-dir", required=True, help="destination outside the repository")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gene-set", action="append", default=[], metavar="NAME=PATH",
        help="a pre-declared gene set; repeatable. Selection must not use target values",
    )
    parser.add_argument(
        "--n-random", type=int, default=200,
        help="size-matched random subsets per gene set. This is the control that "
             "separates effect enrichment from the mechanical effect of fewer genes, "
             "so a gene set cannot be run without it",
    )
    parser.add_argument("--min-spread-ratio", type=float, default=1e-3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = reject_repo_paths(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.gene_set and args.n_random < 1:
        raise SystemExit(
            "--gene-set without --n-random gives a number with nothing to read it "
            "against: a subset of a different size is not comparable to the whole"
        )

    gene_sets: dict[str, set[str]] = {}
    digests: dict[str, str] = {}
    for item in args.gene_set:
        if "=" not in item:
            raise SystemExit(f"--gene-set expects NAME=PATH, got {item!r}")
        name, _, path = item.partition("=")
        gene_sets[name], digests[name] = read_gene_set(Path(path))

    frame, endpoints, level = load_targets(args.prepared)
    floor = OPS_TIER_RULES.reliability_min
    print(json.dumps({
        "aggregation_level": level, "rows": len(frame),
        "screens": int(frame.screen_id.nunique()),
        "genes": int(frame.perturbation_id.nunique()), "endpoints": len(endpoints),
        "reliability_floor": floor,
        "gene_sets": {name: len(members) for name, members in gene_sets.items()},
    }, indent=2), flush=True)

    blocks = split_half_blocks(frame, endpoints, seed=args.seed)
    del frame
    universe = sorted({gene for block in blocks for gene in block.genes})
    print(f"\n{len(universe):,} genes carry two or more sgRNAs in at least one screen")

    def record(name: str, draw: int, declared: int, members: set[str] | None) -> dict:
        by_block = reliability_by_block(blocks, members, min_spread_ratio=args.min_spread_ratio)
        return {
            "gene_set": name, "draw": draw, "n_declared": declared,
            "n_present": len(members) if members is not None else len(universe),
            **summarise_reliability(by_block, floor=floor),
        }

    records = [record("all genes", -1, len(universe), None)]
    print("\nall genes:")
    print(json.dumps({k: v for k, v in records[0].items() if k != "draw"}, indent=2), flush=True)

    for name, members in gene_sets.items():
        present = members.intersection(universe)
        records.append(record(name, -1, len(members), present))
        print(f"\n{name}: {len(present):,} of {len(members):,} declared symbols present")
        print(json.dumps({k: v for k, v in records[-1].items() if k != "draw"}, indent=2), flush=True)

        pool = np.asarray(universe)
        for draw in range(args.n_random):
            rng = np.random.default_rng([args.seed, draw, len(present)])
            sample = set(rng.choice(pool, size=len(present), replace=False))
            records.append(record(f"random matched to {name}", draw, len(present), sample))
        drawn = pd.DataFrame([r for r in records if r["gene_set"] == f"random matched to {name}"])
        for column in ("reliability", "reliability_pooled_native_all_endpoints"):
            low, mid, high = np.nanpercentile(drawn[column], [5, 50, 95])
            observed = records[-len(drawn) - 1][column]
            verdict = "outside" if not low <= observed <= high else "inside"
            print(f"  random control {column}: median {mid:.4f}, 90% interval "
                  f"[{low:.4f}, {high:.4f}] over {len(drawn)} draws; "
                  f"{name} at {observed:.4f} is {verdict} it", flush=True)

    table = pd.DataFrame(records)
    table.to_csv(output / "reliability_audit.csv", index=False)
    (output / "provenance.json").write_text(json.dumps({
        "prepared": str(args.prepared), "reporter_id": REPORTER_ID, "seed": args.seed,
        "aggregation_level": level, "n_random": args.n_random,
        "min_spread_ratio": args.min_spread_ratio, "reliability_floor": floor,
        "gene_set_sha256": digests,
        "note": (
            "reliability standardises each endpoint by the observed spread of the full "
            "gene response before pooling, which is the rule recoverability_by_fold "
            "applies to the recoverability block. reliability_pooled_native_all_endpoints "
            "is what this study reported before 2026-08-15."
        ),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {output / 'reliability_audit.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
