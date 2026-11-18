# MorphoSuff v3 reference checkpoints

These are original fitted study files, not retrained estimates or the full
checkpoints for the manuscript's ensemble.

| Hub file | Fitted setting | Upstream implementation |
| --- | --- | --- |
| `models/scbutterfly_lamp1_ops0011/model.pt` | LAMP1, trained on OPS0012, evaluated on OPS0011, full labels, single state | [scButterfly](https://github.com/BioX-NKU/scButterfly/tree/eb31e04bb8c4abdf85c4cbecd044fcd359105caa), MIT |
| `models/tabm_lamp1_ops0011/model.pt` | LAMP1, trained on OPS0012, evaluated on OPS0011, full labels, single state | [TabM](https://github.com/yandex-research/tabm/tree/28e47ae301c92ec37787dde1ce923a0793f405b4), Apache-2.0 |

The LAMP1 directories each include `preprocessing.json` with fitted
normalization and feature/endpoint order. Use the matching MorphoSuff
adapter rather than treating the weights as interchangeable across targets,
partitions or features. See the
[model runner guide](https://github.com/limengran98/MorphoSuff/tree/main/studies/ops/runners)
and [model source registry](https://github.com/limengran98/MorphoSuff/blob/main/studies/ops/model_sources.yaml).

Use the [Hub file list](https://huggingface.co/datasets/Amanda1998/MorphoSuff/tree/main)
to check availability. The uploaded release manifest records SHA-256 and size
for each checkpoint and preprocessing file. Source data and third-party
implementations retain their original terms.
