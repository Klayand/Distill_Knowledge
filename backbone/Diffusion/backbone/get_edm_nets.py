import torch
from .edm_nets import EDMPrecond


def get_edm_cifar_cond():
    network_kwargs = dict(model_type='SongUNet', embedding_type='positional', encoder_type='standard',
                          decoder_type='standard', channel_mult_noise=1, resample_filter=[1, 1], model_channels=128,
                          channel_mult=[2, 2, 2], augment_dim=9)
    model = EDMPrecond(img_resolution=32, img_channels=3,
                       label_dim=10, **network_kwargs)
    return model


def get_edm_cifar_uncond():
    network_kwargs = dict(model_type='SongUNet', embedding_type='positional', encoder_type='standard',
                          decoder_type='standard', channel_mult_noise=1, resample_filter=[1, 1], model_channels=128,
                          channel_mult=[2, 2, 2], augment_dim=9)
    model = EDMPrecond(img_resolution=32, img_channels=3,
                       label_dim=0, **network_kwargs)
    return model


def get_edm_imagenet_64x64_cond():
    network_kwargs = dict(model_type='DhariwalUNet', model_channels=192, channel_mult=[1, 2, 3, 4])
    model = EDMPrecond(img_resolution=64, img_channels=3,
                       label_dim=1000, **network_kwargs)
    return model
