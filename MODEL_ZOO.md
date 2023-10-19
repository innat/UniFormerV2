# Model Zoo

Some note:

-  `Frame = input_frame x crop x clip`
  - `input_frame` means how many frames are input for model per inference
  - `crop` means spatial crops (e.g., 3 for left/right/center)
  - `clip` means temporal clips (e.g., 4 means repeted sampling four clips with different start indices)

## K710

| Model                       | Frame | Checkpoints    | Config  |
| --------------------------- | ------ | ----- | -------- |
| UniFormerV2-B/16            | 8     | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K710_B16_8x224.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k710/k710_b16_f8x224/config.yaml) |
| UniFormerV2-L/14            | 8     | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K710_L14_8x224.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k710/k710_l14_f8x224/config.yaml) |
| UniFormerV2-L/14@336        | 8     | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K710_L14_8x336.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k710/k710_l14_f8x336/config.yaml) |


## K400

| Model    | Pretraining  | #Frame | Top-1 | Checkpoints | Config   |
| -------------------- | -------- | ------ | ----- | ---- | ----- |
| UniFormerV2-B/16     | CLIP-400M      | 8x3x4  | 84.4  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K400_B16_8x224.h5)  | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k400/k400_b16_f8x224/config.yaml) |
| UniFormerV2-B/16     | CLIP-400M+K710 | 8x3x4  | 85.6  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K400_K710_B16_8x224.h5)  | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k400/k400+k710_b16_f8x224/config.yaml) |
| UniFormerV2-L/14     | CLIP-400M+K710 | 8x3x4  | 88.8  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K400_K710_L14_8x224.h5)  | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k400/k400+k710_l14_f8x224/config.yaml) |
| UniFormerV2-L/14     | CLIP-400M+K710 | 16x3x4 | 89.1  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K400_K710_L14_16x224.h5)  | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k400/k400+k710_l14_f16x224/config.yaml) |
| UniFormerV2-L/14     | CLIP-400M+K710 | 32x3x2 | 89.3  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K400_K710_L14_32x224.h5)  | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k400/k400+k710_l14_f32x224/config.yaml) |
| UniFormerV2-L/14@336 | CLIP-400M+K710 | 32x3x2 | 89.7  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K400_K710_L14_32x336.h5)  | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k400/k400+k710_l14_f32x336/config.yaml) |
| UniFormerV2-L/14@336 | CLIP-400M+K710 | 64x3x2 | 90.0  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K400_K710_L14_64x336.h5)  | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k400/k400+k710_l14_f64x336/config.yaml) | 



## K600

| Model | Pretraining | #Frame | Top-1 | Checkpoints  | Config    |
| ----------- | ------ | ------ | ---- | ------ | ------- |
| UniFormerV2-B/16     | CLIP-400M      | 8x3x4  | 85.0  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K600_B16_8x224.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k600/k600_b16_f8x224/config.yaml) |
| UniFormerV2-B/16     | CLIP-400M+K710 | 8x3x4  | 86.1  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K600_K710_B16_8x224.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k600/k600+k710_b16_f8x224/config.yaml) |
| UniFormerV2-L/14     | CLIP-400M+K710 | 8x3x4  | 89.0  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K600_K710_L14_8x224.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k600/k600+k710_l14_f8x224/config.yaml) |
| UniFormerV2-L/14     | CLIP-400M+K710 | 16x3x4 | 89.4  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K600_K710_L14_16x224.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k600/k600+k710_l14_f16x224/config.yaml) |
| UniFormerV2-L/14     | CLIP-400M+K710 | 32x3x2 | 89.5  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K600_K710_L14_32x224.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k600/k600+k710_l14_f16x224/config.yaml) |
| UniFormerV2-L/14@336 | CLIP-400M+K710 | 32x3x2 | 89.9  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K600_K710_L14_32x336.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k600/k600+k710_l14_f32x336/config.yaml) |
| UniFormerV2-L/14@336 | CLIP-400M+K710 | 64x3x2 | 90.1  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K600_K710_L14_64x336.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k600/k600+k710_l14_f64x336/config.yaml) |


## K700

| Model   | Pretraining    | #Frame | Top-1 | Checkpoints | Config   |
| --------- | ----- | ------ |----- | ------ | ---- |
| UniFormerV2-B/16     | CLIP-400M      | 8x3x4  | 75.8  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K700_B16_8x224.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k700/k700_b16_f8x224/config.yaml) |
| UniFormerV2-B/16     | CLIP-400M+K710 | 8x3x4  | 76.3  | [SavedModel]()/[h5]() | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k700/k700+k710_b16_f8x224/config.yaml) |
| UniFormerV2-L/14     | CLIP-400M+K710 | 8x3x4  | 80.8  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K700_K710_L14_8x224.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k700/k700+k710_l14_f8x224/config.yaml) |
| UniFormerV2-L/14     | CLIP-400M+K710 | 16x3x4 | 81.2  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K700_K710_L14_16x224.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k700/k700+k710_l14_f16x224/config.yaml) |
| UniFormerV2-L/14     | CLIP-400M+K710 | 32x3x2 | 81.5  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K700_K710_L14_32x224.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k700/k700+k710_l14_f32x224/config.yaml) |
| UniFormerV2-L/14@336 | CLIP-400M+K710 | 32x3x2 | 82.1  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K700_K710_L14_32x336.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k700/k700+k710_l14_f32x336/config.yaml) |
| UniFormerV2-L/14@336 | CLIP-400M+K710 | 64x3x2 | 82.7  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_K700_K710_L14_64x336.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/k700/k700+k710_l14_f64x336/config.yaml) |



## Moments in Time V1

| Model  | Pretraining | #Frame | Top-1 | Checkpoints  | Config  |
| -------- | ---------- | ------ | ------ | ------------ | ----------- |
| UniFormerV2-B/16     | CLIP-400M+K710+K400 | 8x3x4  | 42.6  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_MITV1_K710_K400_B16_8x224.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/mit/mit_b16_f8x224/config.yaml) |
| UniFormerV2-L/14     | CLIP-400M+K710+K400 | 8x3x4  | 47.0  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_MITV1_K710_K400_L14_8x224.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/mit/mit_l14_f8x224/config.yaml) |
| UniFormerV2-L/14@336 | CLIP-400M+K710+K400 | 8x3x4  | 47.8  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_MITV1_K710_K400_L14_8x336.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/mit/mit_l14_f8x336/config.yaml) |


## Something-Something V2

| Model | Pretraining | #Frame | Top-1 | Checkpoints  | Config  |
| --- | ----------- | ------ | ----- | -------- | ---------- | 
| UniFormerV2-B/16 | CLIP-400M   | 16x3x1 | 69.5  | [SavedModel]()/[h5]() | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/sthv2/ssv2_b16_f16x224/config.yaml) |
| UniFormerV2-B/16 | CLIP-400M   | 32x3x1 | 70.7  | [SavedModel]()/[h5]() | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/sthv2/ssv2_b16_f32x224/config.yaml) |
| UniFormerV2-L/14 | CLIP-400M   | 16x3x1 | 72.1  | [SavedModel]()/[h5]() | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/sthv2/ssv2_l14_f16x224/config.yaml) |
| UniFormerV2-L/14 | CLIP-400M   | 32x3x1 | 73.0  | [SavedModel]()/[h5]() | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/sthv2/ssv2_l14_f32x224/config.yaml) |

## ActivityNet

| Model  | Pretraining  | #Frame  | Top-1 | Checkpoints  | Config  |
| --- | ----- | ------- | ----- | ---------- | ------ | 
| UniFormerV2-L/14 | CLIP-400M+K710+K400 | 16x3x10 | 94.3  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_ANET_L14_16x224.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/anet/anet_l14_16x224/config.yaml) |
| UniFormerV2-L/14 | CLIP-400M+K710+K400 | 32x3x10 | 94.7  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_ANET_L14_32x224.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/anet/anet_l14_32x224/config.yaml) |

## HACS

| Model | Pretraining  | #Frame  | Top-1 | Checkpoints  | Config  |
| ---------------- | ----- | ------- | ----- | ----------- | ----------- 
| UniFormerV2-L/14 | CLIP-400M+K710+K400 | 16x3x10 | 95.5  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_HACS_L14_16x224.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/hacs/hacs_l14_16x224/config.yaml) |
| UniFormerV2-L/14 | CLIP-400M+K710+K400 | 32x3x10 | 95.4  | [SavedModel]()/[h5](https://github.com/innat/UniFormerV2/releases/download/v1.0/TFUniFormerV2_HACS_L14_32x224.h5) | [cfg](https://github.com/OpenGVLab/UniFormerV2/blob/main/exp/hacs/hacs_l14_32x224/config.yaml) |

