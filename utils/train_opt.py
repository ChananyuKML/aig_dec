import argparse
import os

class TrainOptions():
    def __init__(self):
        self.initialized = False
    
    def initialize(self,parser):
        parser.add_argument('--dataset', default="datset")
        parser.add_argument('--network', default="Res18_2Channels")
        parser.add_argument('--loader', default="gen_image")
        parser.add_argument('--epochs', type=int, default=30)
        parser.add_argument('--lr', type=float, default=1e-5)
        parser.add_argument('--dataroot', type=str, default="dataset/GenImage/sd4")
        parser.add_argument('--batch_size', type=int, default="64")
        parser.add_argument('--transform', default="hogfft_256")
        self.initialized = True
        return parser
    
    def collect_opts(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()
    
    def parse(self):
        opt = self.collect_opts()
        self.opt = opt
        return self.opt