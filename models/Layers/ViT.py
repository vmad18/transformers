from utils.consts import *
from utils.Layers import MultiHeadAttention, FFN


class ViTEncoder(Module):

    def __init__(self, dims: int = 512, h: int = 8, up: int = 2048, dp: int = .1) -> None:
        super().__init__()

        self.msa: MultiHeadAttention = MultiHeadAttention(dims, h, dp)
        self.ffn = FFN(up, dims)

        self.ln1 = nn.LayerNorm(dims, eps=1e-6)
        self.ln2 = nn.LayerNorm(dims, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        skip = x
        x = self.msa(x)
        x = self.ln1(x+skip)
        skip = x
        x = self.ffn(x)
        return self.ln2(x+skip)
