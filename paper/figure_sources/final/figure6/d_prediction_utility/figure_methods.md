# Figure 6b methods

The panel joins coherent checkpoint-average ensemble recoverability to the strict training-reference utility target in Supplementary Figure 4a for all 52 reporters. Recoverability is the arithmetic fivefold mean of endpoint-macro Pearson for the gene×screen elementwise median ensemble. Utility maps each reporter's four components against the other 51 reporters' training references. Figure 6a,c instead use whole-cohort percentiles.

The vertical line is the frozen recovery gate (r=0.70); the horizontal line is the displayed utility median, a descriptive split, not a decision gate. Colour is the five-gate tier; diamonds are unchanged pre-registered domain-sensitivity flags. Six illustrative reporter identities are unchanged; label positions may be adjusted to clear points and gates.

Spearman rho and the two-sided P value use a temporary utility copy rounded to 12 decimals before average ranking. The 95% reporter-bootstrap interval is the same Recovery association as Supplementary Figure 4a (10,000 resamples, seed 20260831); its source is checked against the 52 displayed rows. Coordinates and source values are not rounded. `../b_nested_loo_audit/build_panel_b.py --verify-statistics` verifies this deposited summary.
