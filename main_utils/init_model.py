from models.unet import UNet
from models.unet_do import UNet_DO


def init_model(args):
    _n_classes = 1
    _n_channels = 1
    if args['mode'] == '2.5D':
        _n_channels = 3

    if args['uncertainty_method'] == 'MCDO':
        model = UNet_DO(n_channels=_n_channels, n_classes=_n_classes)
    else:
        model = UNet(n_channels=_n_channels, n_classes=_n_classes)

    return model

