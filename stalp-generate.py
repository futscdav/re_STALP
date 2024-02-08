from futscml_exports import *
import random
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from argparse import ArgumentParser
from stalp import ImageToImageGenerator_JohnsonFutschik

class InferDataset(Dataset):
    def __init__(self, dataroot, xform):
        self.root = dataroot
        self.frames = images_in_directory(self.root)
        self.xform = xform 
            
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        x = pil_loader(os.path.join(self.root, self.frames[idx]))
        x = self.xform(x)
        return x, self.frames[idx]

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument('checkpoint', help='Network checkpoint file to use', type=str)
    p.add_argument('input', help='Input directory to stylize', type=str)
    p.add_argument('output', help='Output directory', type=str)
    p.add_argument('--device', default='cuda:0', type=str)
    args = p.parse_args()

    model = ImageToImageGenerator_JohnsonFutschik(norm_layer='instance_norm', use_bias=True, tanh=True, resnet_blocks=9, append_blocks=True)
    model = torch.jit.script(model)

    ckpt = torch.load(args.checkpoint)['state_dict']
    model.load_state_dict(ckpt)

    model = model.to(args.device).eval()

    transform = ImageTensorConverter(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], resize=None, drop_alpha=True)
    dataset = InferDataset(args.input, transform)
    dataset = DataLoader(dataset, num_workers=0)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for batch in dataset:
        t, p = batch[0], batch[1]
        r = model(t.to(args.device))
        n, c, h, w = r.shape
        for i in range(n):
            transform(r[i]).save(os.path.join(args.output, p[i]))

