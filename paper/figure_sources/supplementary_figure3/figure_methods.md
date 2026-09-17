# Supplementary Figure 3 methods

We reapplied the frozen measurement-sufficiency rules separately to each of the
ten methods using completed held-out-gene predictions, without retraining,
retuning or choosing a method for an individual reporter. Gene-screen keys,
endpoint order, five outer folds, matched controls and observed-target reliability
were held fixed. The primary comparison used checkpoint-average evaluation
outputs where available; a separate single-checkpoint sensitivity used the same
protocol. Ridge, GBDT, CatBoost and MLP had one frozen output shared by both state
labels. Saved training-fold target standard deviations converted each observed
and predicted response to the common within-screen-centred NTC population-SD scale.

Recoverability was the mean over five folds of the equal-endpoint macro-Pearson
correlation across gene-screen responses. For fidelity, response coordinates were
averaged with equal screen weight within each held-out gene, after conversion to
the common scale. Euclidean magnitudes across all 1,000 genes supplied magnitude
Spearman, predicted-to-observed variance ratio and recall of the 50 strongest
observed responses. The coherent ensemble took the element-wise, unweighted
median across ten same-state profiles within fold and gene-screen-endpoint before
screen averaging; every ensemble metric, including recoverability, was evaluated
from those ensemble profiles. This differs from taking the median of ten scalar
recovery scores.

Both states' reference-relative comparisons use the fixed manuscript reference:
the checkpoint-average coherent median ensemble. Reliability below 0.30 determined
nine model-invariant N assignments before any predictive gate. We therefore
reported method-stability summaries for the 43 reliability-passing reporters,
while retaining all 52 in the complete matrix and exact transition tables.
Proxy status denotes Q or R; Q/R changes were kept distinct from crossings to M
or U, and unresolved was not called measurement required. Methods were not
treated as independent biological replicates. Reporter order was unchanged from
Figure 6. LAMP1 was selected from the completed results as a compact example of
the three different outcomes, rather than claimed as a prespecified case.

The figure renderer resolves every plotted key to the canonical Supplementary
Data 1 table and performs no analysis, clustering or threshold fitting. Matrix
cells carry both category letters and the Figure 6 tier colours. Clean standalone
panels share the exact geometry used in the 183 × 162 mm composite, whose letters
are added only during composition. Live PDF/SVG text and 600-dpi PNG are exported.
