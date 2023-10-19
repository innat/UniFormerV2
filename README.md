# UniFormerV2

[](./assets/framework.png)

UniFormerV2, a generic paradigm to build a powerful family of video networks, by arming the pre-trained [**ViTs**](https://github.com/google-research/vision_transformer) with efficient [**UniFormer**](https://github.com/Sense-X/UniFormer) designs. It gets the state-of-the-art recognition performance on **8** popular video benchmarks, including scene-related Kinetics-400/600/700 and Moments in Time, temporal-related Something-Something V1/V2, untrimmed ActivityNet and HACS. In particular, **it is the first model to achieve 90% top-1 accuracy on Kinetics-400.**

This is unofficial `keras` implementation of [**UniFormerV2: Spatiotemporal Learning by Arming Image ViTs with Video UniFormer.**](https://arxiv.org/abs/2211.09552). The official PyTorch code is [here](https://github.com/OpenGVLab/UniFormerV2).


## News

- ?


# Install

```python
git clone https://github.com/innat/UniFormerV2.git
cd UniFormerV2
pip install -e . 
```

# Usage

The `UniFormerV2` checkpoints are available in both `SavedModel` and `H5` formats on total **8** datasets, i.e. [Kinetics-400/600/700/710](https://www.deepmind.com/open-source/kinetics), [Something Something V2](https://developer.qualcomm.com/software/ai-datasets/something-something), [Moments in Time V1](http://moments.csail.mit.edu/), [ActivityNet](http://activity-net.org/) and [HACS](http://hacs.csail.mit.edu/). The variants of this models are `base` and `large`. Each variants may have further variation for different number of input size and input frame. That gives around **32** checkpoints for UniFormerV2. Check this [release](https://github.com/innat/UniFormerV2/releases/tag/v1.0) and [model zoo](MODEL_ZOO.md) page to know details of it. Also check [`model_configs.py`](./model_configs.py) to get overall looks of avaiable model config. Following are some hightlights.

**Inference**

```python
from uniformerv2 import UniFormerV2

>>> model = UniFormerV2(name='K400_B16_8x224')
>>> model.load_weights('TFUniFormerV2_K400_B16_8x224.h5')
>>> container = read_video('sample.mp4')
>>> frames = frame_sampling(container, num_frames=8)
>>> y = model(frames)
>>> y.shape
TensorShape([1, 400])

>>> probabilities = tf.nn.softmax(y_pred_tf)
>>> probabilities = probabilities.numpy().squeeze(0)
>>> confidences = {
    label_map_inv[i]: float(probabilities[i]) \
    for i in np.argsort(probabilities)[::-1]
}
>>> confidences
```
