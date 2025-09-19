import os
import argparse
import torch

from videogpt import VideoData, VideoGPT, load_videogpt
from videogpt.utils import save_video_grid, get_device


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='ucf101_uncond_gpt')
parser.add_argument('--n', type=int, default=8)
parser.add_argument('--device', type=str, default=None, 
                    help='Device to use: cuda, mps, cpu, or auto (default: auto)')
args = parser.parse_args()
n = args.n

# Get the appropriate device
device = get_device(args.device)
print(f"Using device: {device}")

if not os.path.exists(args.ckpt):
    gpt = load_videogpt(args.ckpt, device=device)
else:
    gpt = VideoGPT.load_from_checkpoint(args.ckpt).to(device)
gpt.eval()
args = gpt.hparams['args']

args.batch_size = n
data = VideoData(args)
loader = data.test_dataloader()
batch = next(iter(loader))
batch = {k: v.to(device) for k, v in batch.items()}

samples = gpt.sample(n, batch)
save_video_grid(samples, 'samples.mp4')
