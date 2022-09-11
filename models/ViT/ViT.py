from utils.consts import *
from utils.Layers import PatchEmbedding, ClassificationHead
from models.Layers.ViT import ViTEncoder


'''
Implementation of the Vision Transformer (an image is worth 16x16 words).
'''


class ViT(Module):

    def __init__(self, classes: int, b: int = 6, ed: int = 512, h: int = 8, up: int = 2048, dp: int = .1) -> None:
        super().__init__()

        self.embeds: PatchEmbedding = PatchEmbedding()
        self.blocks: list = [ViTEncoder(ed, h, up, dp) for _ in range(b)]
        self.clh: ClassificationHead = ClassificationHead(ed, classes)

    def forward(self, x: Tensor):
        x = self.embeds(x)
        for b in self.blocks:
            x = b(x)
        return self.clh(x)
