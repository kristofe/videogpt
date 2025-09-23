import os
import itertools
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl

from .resnet import resnet34
from .attention import AttentionStack, LayerNorm, AddBroadcastPosEmbed
from .utils import shift_dim


class VideoGPT(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load VQ-VAE and set all parameters to no grad
        from .vqvae import VQVAE
        from .download import load_vqvae
        if not os.path.exists(args.vqvae):
            self.vqvae = load_vqvae(args.vqvae)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(args.vqvae)
        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.codebook._need_init = False
        self.vqvae.eval()

        # ResNet34 for frame conditioning
        self.use_frame_cond = args.n_cond_frames > 0
        if self.use_frame_cond:
            frame_cond_shape = (args.n_cond_frames,
                                args.resolution // 4,
                                args.resolution // 4,
                                240)
            self.resnet = resnet34(1, (1, 4, 4), resnet_dim=240)
            self.cond_pos_embd = AddBroadcastPosEmbed(
                shape=frame_cond_shape[:-1], embd_dim=frame_cond_shape[-1]
            )
        else:
            frame_cond_shape = None

        # VideoGPT transformer
        self.shape = self.vqvae.latent_shape

        self.fc_in = nn.Linear(self.vqvae.embedding_dim, args.hidden_dim, bias=False)
        self.fc_in.weight.data.normal_(std=0.02)

        self.attn_stack = AttentionStack(
            self.shape, args.hidden_dim, args.heads, args.layers, args.dropout,
            args.attn_type, args.attn_dropout, args.class_cond_dim, frame_cond_shape
        )

        self.norm = LayerNorm(args.hidden_dim, args.class_cond_dim)

        self.fc_out = nn.Linear(args.hidden_dim, self.vqvae.n_codes, bias=False)
        self.fc_out.weight.data.copy_(torch.zeros(self.vqvae.n_codes, args.hidden_dim))

        # caches for faster decoding (if necessary)
        self.frame_cond_cache = None

        self.save_hyperparameters()

    def get_reconstruction(self, videos):
        return self.vqvae.decode(self.vqvae.encode(videos))

    def sample(self, n, batch=None):
        device = self.fc_in.weight.device

        cond = dict()
        if self.use_frame_cond or self.args.class_cond:
            assert batch is not None
            video = batch['video']

            if self.args.class_cond:
                label = batch['label']
                cond['class_cond'] = F.one_hot(label, self.args.class_cond_dim).type_as(video)
            if self.use_frame_cond:
                cond['frame_cond'] = video[:, :, :self.args.n_cond_frames]

        samples = torch.zeros((n,) + self.shape).long().to(device)
        idxs = list(itertools.product(*[range(s) for s in self.shape]))

        with torch.no_grad():
            prev_idx = None
            for i, idx in enumerate(tqdm(idxs)):
                batch_idx_slice = (slice(None, None), *[slice(i, i + 1) for i in idx])
                batch_idx = (slice(None, None), *idx)
                embeddings = self.vqvae.codebook.dictionary_lookup(samples)

                if prev_idx is None:
                    # set arbitrary input values for the first token
                    # does not matter what value since it will be shifted anyways
                    embeddings_slice = embeddings[batch_idx_slice]
                    samples_slice = samples[batch_idx_slice]
                else:
                    embeddings_slice = embeddings[prev_idx]
                    samples_slice = samples[prev_idx]

                logits = self(embeddings_slice, samples_slice, cond,
                              decode_step=i, decode_idx=idx)[1]
                # squeeze all possible dim except batch dimension
                logits = logits.squeeze().unsqueeze(0) if logits.shape[0] == 1 else logits.squeeze()
                probs = F.softmax(logits, dim=-1)
                samples[batch_idx] = torch.multinomial(probs, 1).squeeze(-1)

                prev_idx = batch_idx_slice
            samples = self.vqvae.decode(samples)
            samples = torch.clamp(samples, -0.5, 0.5) + 0.5

        return samples # BCTHW in [0, 1]


    def forward(self, x, targets, cond, decode_step=None, decode_idx=None):
        if self.use_frame_cond:
            if decode_step is None:
                cond['frame_cond'] = self.cond_pos_embd(self.resnet(cond['frame_cond']))
            elif decode_step == 0:
                self.frame_cond_cache = self.cond_pos_embd(self.resnet(cond['frame_cond']))
                cond['frame_cond'] = self.frame_cond_cache
            else:
                cond['frame_cond'] = self.frame_cond_cache

        h = self.fc_in(x)
        h = self.attn_stack(h, cond, decode_step, decode_idx)
        h = self.norm(h, cond)
        logits = self.fc_out(h)

        loss = F.cross_entropy(shift_dim(logits, -1, 1), targets)

        return loss, logits

    def training_step(self, batch, batch_idx):
        self.vqvae.eval()
        x = batch['video']

        cond = dict()
        if self.args.class_cond:
            label = batch['label']
            cond['class_cond'] = F.one_hot(label, self.args.class_cond_dim).type_as(x)
        if self.use_frame_cond:
            cond['frame_cond'] = x[:, :, :self.args.n_cond_frames]

        with torch.no_grad():
            targets, x = self.vqvae.encode(x, include_embeddings=True)
            x = shift_dim(x, 1, -1)

        loss, _ = self(x, targets, cond)
        
        # Log training loss
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Generate and save video predictions every 100 steps
        if batch_idx % self.args.save_video_every == 0:
            self._save_video_predictions(batch, 'train')
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('val/loss', loss, prog_bar=True)
        
        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('val/learning_rate', current_lr, prog_bar=True)
        
        # Generate and save video predictions for validation only occasionally
        # Use save_val_video_every parameter to control frequency (default: only first batch)
        #if batch_idx % self.args.save_val_video_every == 0:
        #    self._save_video_predictions(batch, 'val')

    def _save_video_predictions(self, batch, prefix):
        """Generate and save video predictions as animated GIFs."""
        try:
            import os
            import imageio
            import numpy as np
            
            # Create output directory
            output_dir = f"video_predictions/{prefix}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Take the first sample from the batch
            x = batch['video'][:1]  # [1, C, T, H, W]
            
            # Create a batch for sampling (only first sample)
            sample_batch = {'video': x}
            if self.args.class_cond and 'label' in batch:
                sample_batch['label'] = batch['label'][:1]
            
            # Generate prediction
            with torch.no_grad():
                # Generate video prediction
                predicted_video = self.sample(1, sample_batch)
                
                # Convert to numpy and ensure proper range
                original_video = x[0].cpu().numpy()  # [C, T, H, W]
                predicted_video = predicted_video[0].cpu().numpy()  # [C, T, H, W]
                
                # Convert from [0, 1] to [0, 255] range
                original_video = np.clip(original_video * 255, 0, 255).astype(np.uint8)
                predicted_video = np.clip(predicted_video * 255, 0, 255).astype(np.uint8)
                
                # Convert from [C, T, H, W] to [T, H, W, C] for imageio
                if original_video.shape[0] == 3:  # RGB
                    original_video = original_video.transpose(1, 2, 3, 0)  # [T, H, W, C]
                    predicted_video = predicted_video.transpose(1, 2, 3, 0)  # [T, H, W, C]
                else:  # Grayscale
                    original_video = original_video[0].transpose(1, 2, 0)  # [T, H, W, 1]
                    predicted_video = predicted_video[0].transpose(1, 2, 0)  # [T, H, W, 1]
                    # Convert grayscale to RGB for GIF
                    original_video = np.repeat(original_video, 3, axis=-1)
                    predicted_video = np.repeat(predicted_video, 3, axis=-1)
                
                # Create side-by-side comparison
                side_by_side = np.concatenate([original_video, predicted_video], axis=2)  # [T, H, 2*W, C]
                
                # Save as animated GIF
                filename = f"{output_dir}/step_{self.global_step:06d}.gif"
                imageio.mimsave(filename, side_by_side, duration=0.2, loop=0)
                
                # Also save original and predicted separately
                original_filename = f"{output_dir}/original_step_{self.global_step:06d}.gif"
                predicted_filename = f"{output_dir}/predicted_step_{self.global_step:06d}.gif"
                
                imageio.mimsave(original_filename, original_video, duration=0.2, loop=0)
                imageio.mimsave(predicted_filename, predicted_video, duration=0.2, loop=0)
                
                # Add videos directly to TensorBoard
                try:
                    # Convert back to tensor format for TensorBoard: [N, T, C, H, W]
                    original_tensor = torch.from_numpy(original_video).permute(0, 3, 1, 2).unsqueeze(0)  # [1, T, C, H, W]
                    predicted_tensor = torch.from_numpy(predicted_video).permute(0, 3, 1, 2).unsqueeze(0)  # [1, T, C, H, W]
                    side_by_side_tensor = torch.from_numpy(side_by_side).permute(0, 3, 1, 2).unsqueeze(0)  # [1, T, C, H, W]
                    
                    # Add videos to TensorBoard
                    self.logger.experiment.add_video(f'{prefix}/original_video', original_tensor, self.global_step, fps=5)
                    self.logger.experiment.add_video(f'{prefix}/predicted_video', predicted_tensor, self.global_step, fps=5)
                    self.logger.experiment.add_video(f'{prefix}/side_by_side', side_by_side_tensor, self.global_step, fps=5)
                    
                except Exception as e:
                    print(f"Warning: Failed to add videos to TensorBoard: {e}")
                
                # Log file paths to TensorBoard
                self.logger.experiment.add_text(f'{prefix}/video_prediction_path', filename, self.global_step)
                
        except Exception as e:
            # If video prediction fails, just skip it to avoid breaking training
            if self.global_step % 100 == 0:
                print(f"Warning: Failed to save video predictions: {e}")
            pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
        return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--vqvae', type=str, default='kinetics_stride4x4x4',
                            help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--n_cond_frames', type=int, default=0)
        parser.add_argument('--class_cond', action='store_true')

        # VideoGPT hyperparmeters
        parser.add_argument('--hidden_dim', type=int, default=576)
        parser.add_argument('--heads', type=int, default=4)
        parser.add_argument('--layers', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--attn_type', type=str, default='full',
                            choices=['full', 'sparse'])
        parser.add_argument('--attn_dropout', type=float, default=0.3)
        parser.add_argument('--save_video_every', type=int, default=1000,
                            help='Save video predictions every N training steps')
        parser.add_argument('--save_val_video_every', type=int, default=1,
                            help='Save video predictions every N validation batches (default: 1, only first batch)')

        return parser
