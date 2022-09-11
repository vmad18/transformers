from utils.consts import *
from models.Layers.Swin import SwinLayer

'''
Implementation of the Shifted Window Transformer
'''


class SwinTransformer(Module):

    """
    :param classes - number of classes
    :param b - number of Swin Blocks per stage
    :param dim - initial dimension
    :param resolution - input resolution/image dims
    :param ps - parition size
    :param heads - number of SWMSA heads
    :param dp - Dropout value
    :param ws - Window Size
    """

    def __init__(self, classes: int = 10, b: tuple[int] = [2, 2, 6, 2], dim: int = 48, resolution: tuple = (224, 224), ps: int = 4, heads: tuple[int] = (3, 6, 12, 24), dp: int = .1, ws: int = 7):
        super().__init__()
        self.layers = nn.ModuleList([SwinLayer(n, dim*(2**i), (resolution[0]//(ps * 2**i), resolution[1]//(ps * 2**i)), ps if i == 0 else null, heads[i], dp, ws, true) for i, n in enumerate(b)])

        self.squish = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(dim*(2**(len(b)-1)), classes)

    def forward(self, x: Tensor) -> Tensor:

        for layer in self.layers:
            x = layer(x)

        x = x.transpose(1, 2)
        x = self.squish(x).squeeze(2)
        return F.softmax(self.classifier(x), -1)


if __name__ == "__main__":
    tnsr: Tensor = torch.randn((3, 3, 224, 224))
    swin: SwinTransformer = SwinTransformer()
