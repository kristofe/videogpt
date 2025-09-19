import os
import functools
import argparse
from videogpt.download import load_i3d_pretrained
from tqdm import tqdm
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from videogpt.fvd.fvd import get_fvd_logits, frechet_distance
from videogpt import VideoData, VideoGPT, load_videogpt
from videogpt.utils import get_device, get_device_count, is_distributed_available


MAX_BATCH = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='bair_gpt')
    parser.add_argument('--n_trials', type=int, default=1, help="Number of trials to compute mean/std")
    parser.add_argument('--port', type=int, default=23452)
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use: cuda, mps, cpu, or auto (default: auto)')
    args = parser.parse_args()
    
    # Get the appropriate device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    if device.type == 'cuda' and is_distributed_available():
        ngpus = get_device_count()
        assert 256 % ngpus == 0, f"Must have 256 % n_gpus == 0"
        mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, args), join=True)
    else:
        # For MPS and CPU, run single process
        main_worker(0, 1, args)


def main_worker(rank, size, args_in):
    global args
    args = args_in
    is_root = rank == 0
    
    # Get the appropriate device
    device = get_device(args.device)
    if device.type == 'cuda' and size > 1:
        dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{args.port}',
                                world_size=size, rank=rank)
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        # For MPS and CPU, or single GPU, use the detected device
        device = device
    
    torch.set_grad_enabled(False)

    n_trials = args.n_trials

    #################### Load VideoGPT ########################################
    if not os.path.exists(args.ckpt):
        gpt = load_videogpt(args.ckpt, device=device)
    else:
        gpt = VideoGPT.load_from_checkpoint(args.ckpt).to(device)
    gpt.eval()
    args = gpt.hparams['args']

    if device.type == 'cuda' and size > 1:
        args.batch_size = 256 // dist.get_world_size()
    else:
        args.batch_size = 256
    loader = VideoData(args).test_dataloader()

    #################### Load I3D ########################################
    i3d = load_i3d_pretrained(device)

    #################### Compute FVD ###############################
    fvds = []
    fvds_star = []
    if is_root:
        pbar = tqdm(total=n_trials)
    for _ in range(n_trials):
        fvd, fvd_star = eval_fvd(i3d, gpt, loader, device)
        fvds.append(fvd)
        fvds_star.append(fvd_star)

        if is_root:
            pbar.update(1)
            fvd_mean = np.mean(fvds)
            fvd_std = np.std(fvds)

            fvd_star_mean = np.mean(fvds_star)
            fvd_star_std = np.std(fvds_star)

            pbar.set_description(f"FVD {fvd_mean:.2f} +/- {fvd_std:.2f}, FVD* {fvd_star_mean:.2f} +/0 {fvd_star_std:.2f}")
    if is_root:
        pbar.close()
        print(f"Final FVD {fvd_mean:.2f} +/- {fvd_std:.2f}, FVD* {fvd_star_mean:.2f} +/- {fvd_star_std:.2f}")


def all_gather(tensor):
    if dist.is_initialized():
        rank, size = dist.get_rank(), dist.get_world_size()
        tensor_list = [torch.zeros_like(tensor) for _ in range(size)]
        dist.all_gather(tensor_list, tensor)
        return torch.cat(tensor_list)
    else:
        # For single process (MPS/CPU), just return the tensor
        return tensor


def eval_fvd(i3d, videogpt, loader, device):
    if dist.is_initialized():
        rank, size = dist.get_rank(), dist.get_world_size()
        is_root = rank == 0
    else:
        rank, size = 0, 1
        is_root = True

    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    fake_embeddings = []
    for i in range(0, batch['video'].shape[0], MAX_BATCH):
        fake = videogpt.sample(MAX_BATCH, {k: v[i:i+MAX_BATCH] for k, v in batch.items()})
        fake = fake.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
        fake = (fake * 255).astype('uint8')
        fake_embeddings.append(get_fvd_logits(fake, i3d=i3d, device=device))
    fake_embeddings = torch.cat(fake_embeddings)

    real = batch['video'].to(device)
    real_recon_embeddings = []
    for i in range(0, batch['video'].shape[0], MAX_BATCH):
        real_recon = (videogpt.get_reconstruction(batch['video'][i:i+MAX_BATCH]) + 0.5).clamp(0, 1)
        real_recon = real_recon.permute(0, 2, 3, 4, 1).cpu().numpy()
        real_recon = (real_recon * 255).astype('uint8')
        real_recon_embeddings.append(get_fvd_logits(real_recon, i3d=i3d, device=device))
    real_recon_embeddings = torch.cat(real_recon_embeddings)

    real = real + 0.5
    real = real.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
    real = (real * 255).astype('uint8')
    real_embeddings = get_fvd_logits(real, i3d=i3d, device=device)

    fake_embeddings = all_gather(fake_embeddings)
    real_recon_embeddings = all_gather(real_recon_embeddings)
    real_embeddings = all_gather(real_embeddings)

    expected_batch_size = 256 if size == 1 else 256 // size
    assert fake_embeddings.shape[0] == real_recon_embeddings.shape[0] == real_embeddings.shape[0] == expected_batch_size

    fvd = frechet_distance(fake_embeddings.clone(), real_embeddings)
    fvd_star = frechet_distance(fake_embeddings.clone(), real_recon_embeddings)
    return fvd.item(), fvd_star.item()


if __name__ == '__main__':
    main()
