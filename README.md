# UniFormerV2

Keras re-implementation of [**UniFormerV2: Spatiotemporal Learning by Arming Image ViTs with Video UniFormer".**](https://arxiv.org/abs/2211.09552). The official PyTorch code is [here](https://github.com/OpenGVLab/UniFormerV2).

![](./assets/framework.png)

In UniFormerV2, a generic paradigm to build a powerful family of video networks, by arming the pre-trained ViTs with efficient UniFormer designs. It gets the state-of-the-art recognition performance on 8 popular video benchmarks, including scene-related Kinetics-400/600/700 and Moments in Time, temporal-related Something-Something V1/V2, untrimmed ActivityNet and HACS. In particular, **it is the first model to achieve 90% top-1 accuracy on Kinetics-400.**

# Install

```python

git clone https://github.com/innat/UniFormerV2.git
cd UniFormerV2
pip install -e . 
```
