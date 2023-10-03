# UniFormerV2

Keras re-implementation of [**UniFormerV2: Spatiotemporal Learning by Arming Image ViTs with Video UniFormer".**](https://arxiv.org/abs/2211.09552). The official PyTorch code is [here](https://github.com/OpenGVLab/UniFormerV2).

![]('./assets/framework.png')

In UniFormerV2, a generic paradigm to build a powerful family of video networks, by arming the pre-trained ViTs with efficient UniFormer designs. It gets the state-of-the-art recognition performance on 8 popular video benchmarks, including scene-related Kinetics-400/600/700 and Moments in Time, temporal-related Something-Something V1/V2, untrimmed ActivityNet and HACS. In particular, **it is the first model to achieve 90% top-1 accuracy on Kinetics-400.**

```python
num_frames = 8
num_classes = 174
input_size = 224

def tf_uniformerv2_b16(
    t_size=16, dw_reduction=1.5, backbone_drop_path_rate=0., 
    temporal_downsample=True,
    no_lmhra=False, double_lmhra=True,
    return_list=[8, 9, 10, 11], 
    n_layers=4, n_dim=768, n_head=12, mlp_factor=4.0, drop_path_rate=0.,
    mlp_dropout=[0.5, 0.5, 0.5, 0.5], 
    cls_dropout=0.5, num_classes=400, input_resolution=224,
    name='TF_UniformerV2_B16'
):
    model = TFVisionTransformer(
        input_resolution=input_resolution,
        patch_size=16,
        width=768,
        vit_layers=12,
        heads=12,
        output_dim=512,
        t_size=t_size,
        dw_reduction=dw_reduction, 
        backbone_drop_path_rate=backbone_drop_path_rate,
        temporal_downsample=temporal_downsample,
        no_lmhra=no_lmhra,
        double_lmhra=double_lmhra,
        return_list=return_list, 
        n_layers=n_layers, 
        n_dim=n_dim, 
        n_head=n_head, 
        mlp_factor=mlp_factor, 
        drop_path_rate=drop_path_rate, 
        mlp_dropout=mlp_dropout, 
        cls_dropout=cls_dropout, 
        num_classes=num_classes,
        name=name
    )
    return model


model = tf_uniformerv2_b16(
    t_size=num_frames,
    n_layers=n_layers,
    input_resolution=input_size,
    num_classes=num_classes
)
model.load_weight('TFUniFormerV2_SSV2_B16_16x224.h5')
```
