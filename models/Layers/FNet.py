from utils.consts import *
from utils.Layers import FFN


class FourierMixer(Module):

    """
    Use with GPU
    """

    def __init__(self) -> None:
        super().__init__()

    """
    performs fourier transform along dims and sequence
    :param x - input tensor (bs, tokens, dims)
    :return real component of fourier transforms
    """

    def forward(self, x: Tensor) -> Tensor:
        x = fft(fft(x, dim=2), dim=1)
        return x.real


class FourierMixerMM(Module):

    """
    Use with smaller max token sizes and TPU
    :param dim - embedding dimension
    :param toks - max token size
    """

    def __init__(self, dim: int, toks: int):
        super().__init__()
        self.d_fft = torch.tensor(linalg.dft(dim)) # (dim, dim)
        self.s_fft = torch.tensor(linalg.dft(toks)) # (toks, toks)

    def forward(self, x: Tensor) -> Tensor:
        x = x.type(torch.complex128)
        x = x @ self.d_fft
        x = x.transpose(-1, -2) @ self.s_fft
        return x.transpose(-1, -2).real


class FourierBlock(Module):

    def __init__(self, dp: int, up: int, dim: int) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dp)

        self.fm = FourierMixer()
        self.ffn = FFN(up, dim)

        self.l1 = nn.LayerNorm(dim, eps=1e-6)
        self.l2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        skip = x
        x = self.fm(x)
        x = self.l1(x + skip)
        skip = x
        x = self.ffn(x)
        x = self.l2(x + skip)
        return x


if __name__ == "__main__":
    tnsr: Tensor = torch.randn((1, 1024, 512))
