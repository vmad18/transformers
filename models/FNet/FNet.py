from utils.consts import *
from models.Layers.FNet import FourierBlock


'''
FNet is purely a transformer encoder that replaces multi-head attention with fourier transforms along 
the embedding dimension and sequence dimension. 
'''


class FNet(Module):

    def __init__(self, b: int = 6, ed: int = 512, h: int = 8, up: int = 2048, dp: int = .1):
        super().__init__()

        self.blcks = [FourierBlock(dp, up, ed) for _ in range(b)]
