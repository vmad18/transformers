from utils.consts import *
from models.Swin.Swin import SwinTransformer
'''
from models.ViT.ViT import ViT
from models.FNet.FNet import FNet
'''


def main(tnsr: Tensor) -> null:
    swint: SwinTransformer = SwinTransformer(5)
    print(swint(tnsr).shape)


if __name__ == "__main__":
    tnsr: Tensor = torch.randn((3, 3, 224, 224))
    main(tnsr)
