# Figure source validation

`configs/ops/figures/figure1.yaml` through `figure6.yaml` are portable source
contracts, translated from the frozen display package.  They identify canonical
table names, schema requirements, exclusions, statistical units, consensus
semantics and output expectations. The current publication source packages are
documented in [the figure index](../../../paper/figure_sources/README.md).

Figure rendering must fail if required inputs are absent, duplicate against a
declared key, use a non-registry method, or violate the declared consensus.
Builders must not train or re-estimate benchmark quantities.
