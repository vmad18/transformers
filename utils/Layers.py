from utils.consts import *


class PatchEmbedding(Module):

    def __init__(self, ed: int = 512, ps: int = 16, in_dim: int = 224) -> None:
        super().__init__()

        self.ed = ed
        self.ps = ps
        self.patches = in_dim // self.ps

        self.proj = nn.Conv2d(3, self.ed, kernel_size=self.ps, stride=self.ps)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.ed))  # classification patch
        self.pos_embed = nn.Parameter(
            torch.randn(self.patches ** 2 + 1, self.ed))  # (patches + class token, embedding dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x).permute(0, 2, 3, 1).view(-1, self.patches * self.patches, self.ed) # collapse dimensions (bs, patches * patches, embedding dim)
        cls = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([cls, x], dim=1)
        x += self.pos_embed
        return x


class LocalPositionEmbedding(Module):

    """
    :param md - max distance
    """

    def __init__(self, md: int):
        super().__init__()

        self.md = md

        self.param_dists = nn.Parameter(torch.randn(2*md-1, 2*md-1), requires_grad=true)
        self.pos = torch.tensor(np.asarray([[p1, p2] for p1 in range(md) for p2 in range(md)]))
        self.dist_table = self.pos.unsqueeze(0) - self.pos.unsqueeze(1) + (md - 1)

    def forward(self, x: Tensor) -> Tensor:
        x += self.param_dists[self.dist_table[:, :, 0].long(), self.dist_table[:, :, 1].long()] # works without .long() sometimes
        return x


#TODO fix this
class RelativePositionEmbedding(nn.Module):

    def __init__(self, heads, toks: int):
        super().__init__()
        self.heads = heads
        self.toks = toks
        self.pos_embed = nn.Parameter(torch.randn(heads, toks * 2 - 1))

    def forward(self) -> Tensor:
        dists = torch.arange(self.toks)
        dists = dists.unsqueeze(0) - dists.unsqueeze(1) + self.max_relative_position-1
        embeds = self.pos_embed[:, dists]
        return embeds


class MultiHeadAttention(Module):

    """
    Performs self-attention (cosine similarity) across a set of heads.

    :param dims - Embedding/Hidden Dimension
    :param heads - heads
    :param dp - Dropout percentage
    """

    def __init__(self, dims: int, heads: int, dp: int) -> None:
        super().__init__()

        self.h = heads
        self.dims = dims
        self.sh = self.dim // self.h

        self.drop = nn.Dropout(dp)
        self.qkv = nn.Linear(self.dims, 3 * self.dims)
        self.proj = nn.Linear(self.dims, self.dims)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        wqkv = self.qkv(x).view(x.shape[0], x.shape[1], self.h, 3*self.sh).view(x.shape[0], x.shape[1], self.h, 3*self.sh).permute(0, 2, 1, 3).chunk(3, dim=-1)  # (qkv, bs, heads, seq, ed//3)

        wq, wk, wv = wqkv
        attn: Tensor = (wq @ wk.transpose(-2, -1)) / np.sqrt(self.sh)  # (bs, heads, seq, seq)

        if not (mask is None):
            attn += mask * -1e9

        attn = self.drop(torch.softmax(attn, dim=-1))
        attn = attn @ wv
        attn = attn.permute(0, 2, 1, 3)
        return self.proj(attn.reshape(attn.shape[0], attn.shape[1], attn.shape[2] * attn.shape[3]))


class LocalAttention(Module):

    """
    Performs self-attention across an mxm region.

    :param dims - input dimension
    :param heads - number of heads
    :param ws - window size of local attention
    """

    def __init__(self, dims: int, heads: int, ws: int):
        super().__init__()

        self.pd = dims // heads
        self.h = heads
        self.ws = ws

        self.qkv = nn.Linear(dims, 3*dims)
        self.proj = nn.Linear(dims, dims)

        self.pos_embed = LocalPositionEmbedding(self.ws)

    def forward(self, x: Tensor, mask: Tensor = null) -> Tensor:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)

        x = self.qkv(x).view(B, H, W, self.h, 3*self.pd).permute(0, 3, 1, 2, 4)
        x = x.view(B, self.h, H//self.ws, self.ws, W//self.ws, self.ws, 3*self.pd)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, self.h, H//self.ws * W//self.ws, self.ws*self.ws, 3*self.pd).chunk(3, -1)

        wq, wk, wv = x
        attn = (wq @ wk.transpose(-2, -1)) / np.sqrt(self.pd)

        if mask != null: attn += mask * -1e9

        attn = self.pos_embed(attn)
        attn = F.softmax(attn, -1)
        attn = attn @ wv

        attn = attn.view(B, self.h, H//self.ws, W//self.ws, self.ws, self.ws, self.pd).permute(0, 2, 4, 3, 5, 1, 6)
        attn = attn.reshape(B, H, W, C)
        return self.proj(attn).permute(0, 3, 1, 2)


class FFN(Module):

    """
    Similar to 1D convolution
    :param d - up sample value
    :param dim - input dimension
    :param nl - non linearity activation
    """

    def __init__(self, d: int, dim: int, nl = nn.ReLU(inplace=true)) -> None:
        super().__init__()

        self.nl = nl

        self.d1 = nn.Linear(dim, d)
        self.d2 = nn.Linear(d, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.d2(self.nl(self.d1(x)))


class WindowAttention(Module):

    """
    :param h - heads
    :param ed - embedding dimension
    :param dp - drop out rate
    :param ws - window size
    """

    def __init__(self, h: int, ed: int, dp: int, ws: int, resolution: tuple, shift: bool = false) -> None:
        super().__init__()

        self.h = h
        self.ed = ed
        self.ws = ws
        self.resolution = resolution

        self.sh = 3 * self.ed // self.h

        self.wrp = LocalPositionEmbedding(self.ws)
        self.drop = nn.Dropout(dp)
        self.qkv = nn.Linear(self.ed, 3 * self.ed)
        self.proj = nn.Linear(self.ed, self.ed)

        self.shifted = shift

    """
    :param x - input tensor
    :param ws - window size
    :param ss - shift size
    """

    def window_masks(self, x: Tensor):
        B, H, W, C = x.shape

        mask = x
        area: int = 0

        hp = (slice(0, -self.ws), slice(-self.ws, -self.ws//2), slice(-self.ws//2, null))
        wp = (slice(0, -self.ws), slice(-self.ws, -self.ws//2), slice(-self.ws//2, null))

        for h in hp:
            for w in wp:
                mask[:, h, w, :] = area
                area+=1

        mask = mask.view(mask.shape[0], mask.shape[1] // self.ws, self.ws, mask.shape[2] // self.ws, self.ws, mask.shape[3])
        mask = mask.permute(0, 1, 3, 2, 4, 5).contiguous()
        mask = mask.view(mask.shape[0]*mask.shape[1]*mask.shape[2], self.ws, self.ws, C)

        mask = mask.view(-1, self.ws*self.ws)
        mask = mask.unsqueeze(1) - mask.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, float(-1e9)).masked_fill(mask == 0, 0.0)
        return mask

    def forward(self, x: Tensor) -> Tensor:

        B, P, C = x.shape

        x = x.view(B, self.resolution[0], self.resolution[1], C)

        if self.shifted:
            x = torch.roll(x, (-self.ws//2, -self.ws//2), (1, 2))

        x = self.qkv(x).view(x.shape[0], self.resolution[0]//self.ws, self.ws, self.resolution[1]//self.ws, self.ws, self.h, self.sh) # (bs, h, windows, w, windows, dims)
        x = x.permute(0, 5, 1, 3, 2, 4, 6)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3], self.ws*self.ws, self.sh) # (bs, heads, h, w, windows*windows, dims)
        x = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3], self.ws*self.ws, self.sh).chunk(3, dim=-1) # (qkv, bs, heads, h*w, windows*windows, dims)

        wq, wk, wv = x
        attn = (wq @ wk.transpose(3, 4)) / np.sqrt(self.sh)

        if self.shifted:
            attn_mask = self.window_masks(torch.zeros((1, self.resolution[0], self.resolution[1], 1)))
            attn += attn_mask

        attn = self.wrp(attn)

        attn = F.softmax(attn, -1)
        attn = attn @ wv

        attn = attn.view(attn.shape[0], attn.shape[1], self.resolution[0]//self.ws, self.resolution[1]//self.ws, self.ws, self.ws, attn.shape[4])
        attn = attn.permute(0, 2, 4, 3, 5, 1, 6)
        attn = attn.reshape(attn.shape[0], P, C) # (bs, h, w, dims)

        if self.shifted:
            attn = torch.roll(attn, (self.ws//2, self.ws//2), (1, 2))
        return self.proj(attn)


class ClassificationHead(Module):

    def __init__(self, ed: int, classes: int) -> None:
        super().__init__()

        self.ed = ed

        self.nl = nn.Softmax(-1)
        self.ln = nn.LayerNorm(self.ed, eps=1e-6)
        self.classifier = nn.Linear(self.ed, classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.ln(x[:, 0])
        return self.nl(self.classifier(x))


class PatchMerging(Module):

    def __init__(self, dim: int, resolution: tuple, partition: int = None):
        super().__init__()

        self.dim = dim
        self.partition = partition
        self.partition = partition
        self.resolution = resolution

        if self.partition != null:
            self.cv1 = nn.Conv2d(3, self.dim, kernel_size=self.partition, stride=self.partition)
            self.pos = nn.Parameter(torch.randn(self.resolution[0]*self.resolution[1], self.dim))
        else:
            self.proj = nn.Linear(2*self.dim, self.dim, bias=false) # linear transformation

    def forward(self, x: Tensor):
        if self.partition != null:
            x = self.cv1(x)
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], self.dim) + self.pos
            return x
        else:
            x = x.view(x.shape[0], self.resolution[0], 2, self.resolution[1], 2, x.shape[2]).permute(0, 1, 3, 2, 4, 5)
            x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], 2*self.dim)
            return self.proj(x)


class PositionalEncoding(Module):

    """
    Fourier Theory as a means to add positional information to the parallel processing
    of the transformer

    :param dims - embedding/feature dimension of input
    :param max_seq - max sequence length of the input
    :param dp - dropout rate of the output to help the model vary the features extracted
    """

    def __init__(self, dims: int = 512, max_seq: int = 2048, dp: float = .1):
        super().__init__()

        self.dp = nn.Dropout(p=dp)

        pos = torch.arange(0, max_seq).unsqueeze(1)
        vals = torch.pow(1./1e4, torch.arange(0, dims, 2)/dims)

        self.encodes = torch.zeros(max_seq, dims)
        self.encodes[:, ::2] = torch.sin(pos * vals)
        self.encodes[:, 1::2] = torch.cos(pos * vals)

    def forward(self, x: Tensor) -> Tensor:
        return self.dp(x + x[:x.shape[0], :])


if __name__ == "__main__":
    tnsr: Tensor = torch.randn((3, 56, 224, 224))
    la: LocalAttention = LocalAttention(56, 8, 7)
    print(la(tnsr))
    # pm = PatchMerging(48, resolution=(224//4, 224//4), partition=4)
    # pm2 = PatchMerging(96, resolution=(224//8, 224//8))
