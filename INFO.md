## Sampling

From official [source](https://github.com/OpenGVLab/UniFormerV2/blob/main/INSTRUCTIONS.md).

- They adopt **sparse sampling** for all the datasets.
- For those **scene-related** datasets (e.g., Kinetics), they **ONLY** add global UniBlocks.
- For those **temporal-related** datasets (e.g., Sth-Sth), they adopt **ALL** the designs, including local UniBlocks, global UniBlocks and temporal downsampling.

The following terms are used as variable while building the model. Some many not self-explainable and needs to be re-touch (TODO).

```yaml
N_LAYERS: 4  # number of global UniBlocks
MLP_DROPOUT: [0.5, 0.5, 0.5, 0.5]  # dropout for each global UniBlocks
CLS_DROPOUT: 0.5  # dropout for the final classification layer
RETURN_LIST: [8, 9, 10, 11]  # layer index for inserting global UniBlocks
NO_LMHRA: True  # whether adding local MHRA in the local UniBlocks
TEMPORAL_DOWNSAMPLE: False  # whether using temporal downsampling in the patch embedding
FROZEN: False  # whether freeze backbone
```


## Kinetics-710

From official [source](https://github.com/OpenGVLab/UniFormerV2/blob/main/DATASET.md), for Kientics-710, author of `UniFormerV2` merge the training set of Kinetics-400/600/700, and then delete the repeated videos according to Youtube IDs. Note they also remove testing videos from different Kinetics datasets leaked in their combined training set for correctness. As a result, the total number of training videos is reduced from `1.14M` to `0.65M`. Additionally, they merge the action categories in these three Kinetics datasets, which leads to `710` classes in total. Hence, they call this video benchmark `Kinetics-710`. More detailed descriptions can be found in their **Appendix E**. 

In their experiments, we empirically show the effectiveness of our Kinetics-710. For post-pretraining, we simply use `8` input frames and adopt the same hyperparameters as training on the individual Kinetics dataset. After that, no matter how many frames are input (`16`, `32`, or even `64`), we only need **5-epoch finetuning** for more than 1% top-1 accuracy improvement on Kinetics-400/600/700.

> When finetuning the K710-pretrained models, they load the weights of classification layers and map the weight according to the label list. **They have provided the label map in the meta files.**

| Model       | Pretrain | #Frame | K400 | K600 | K700 |
| ----------- | -------- | ------ | ---- | ---- | ---- |
| UniFormerV2-B | CLIP-400M  | 8x3x4  | 84.4 | 85.0 | 75.8 |
| UniFormerV2-B | CLIP-400M+K710  | 8x3x4  | **85.6 (+1.2)** | **86.1 (+1.1)** | **76.3 (+0.5)** |
| UniFormerV2-L | CLIP-400M  | 8x3x4  | 87.7 | 88.0 | 80.3 |
| UniFormerV2-L | CLIP-400M+K710  | 8x3x4  | **88.8 (1.1)** | **89.0 (+1.0)** | **80.8 (+0.5)** |
