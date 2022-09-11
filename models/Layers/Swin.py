from utils.consts import *
from utils.Layers import WindowAttention, LocalPositionEmbedding, FFN, PatchMerging


class SwinBlock(Module):

    """
    :param dims - input hidden dimensions
    :param heads - msa heads
    :param dp - dropout rate
    :param ws - window size
    """

    def __init__(self, dim: int, resolution: tuple, heads: int = 8, dp: int = .1, ws: int = 7, shifted: bool = false):
        super().__init__()

        self.swmsa = WindowAttention(heads, dim, dp, ws, resolution, shifted)
        self.ln1 = nn.LayerNorm(dim, eps=1e-6)
        self.ln2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = FFN(4*dim, dim, nl=nn.GELU())

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.ln1(x)
        x = self.swmsa(x)
        shortcut += x
        x = self.ln2(shortcut)
        return self.mlp(x)+shortcut


class SwinLayer(Module):

    """
    :param b - number of blocks
    :param dim - input of feature dimension
    :param heads - msa heads
    :param dp - dropout rate
    :param ws - window size
    """

    def __init__(self, b: int, dim: int, resolution: tuple, ps: int = null, heads: int = 8, dp: int = .1, ws: int = 7, show: bool = false):
        super().__init__()

        self.show = show
        self.pm = PatchMerging(dim, resolution, ps)
        self.blocks = nn.ModuleList([SwinBlock(dim, resolution, heads, dp, ws, i%2 != 0) for i in range(b)])

    def forward(self, x: Tensor) -> Tensor:
        x = self.pm(x)
        if self.show: print(x.shape)
        for block in self.blocks: x = block(x)
        return x
