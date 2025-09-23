import math
import argparse
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .attention import MultiHeadAttention
from .utils import shift_dim

class VQVAE(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding_dim = args.embedding_dim
        self.n_codes = args.n_codes

        self.encoder = Encoder(args.n_hiddens, args.n_res_layers, args.downsample)
        self.decoder = Decoder(args.n_hiddens, args.n_res_layers, args.downsample)

        self.pre_vq_conv = SamePadConv3d(args.n_hiddens, args.embedding_dim, 1)
        self.post_vq_conv = SamePadConv3d(args.embedding_dim, args.n_hiddens, 1)

        self.codebook = Codebook(args.n_codes, args.embedding_dim)
        self.save_hyperparameters()

    @property
    def latent_shape(self):
        input_shape = (self.args.sequence_length, self.args.resolution,
                       self.args.resolution)
        return tuple([s // d for s, d in zip(input_shape,
                                             self.args.downsample)])

    def encode(self, x, include_embeddings=False):
        h = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(h)
        if include_embeddings:
            return vq_output['encodings'], vq_output['embeddings']
        else:
            return vq_output['encodings']

    def decode(self, encodings):
        h = F.embedding(encodings, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h)

    def forward(self, x):
        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']))
        recon_loss = F.mse_loss(x_recon, x) / 0.06

        return recon_loss, x_recon, vq_output

    def training_step(self, batch, batch_idx):
        x = batch['video']
        recon_loss, x_recon, vq_output = self.forward(x)
        commitment_loss = vq_output['commitment_loss']
        loss = recon_loss + commitment_loss
        
        # Log training losses
        self.log('train/recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/commitment_loss', commitment_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/total_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/perplexity', vq_output['perplexity'], on_step=True, on_epoch=True, prog_bar=True)
        
        # Log images every 100 steps
        if batch_idx % 100 == 0:
            self._log_images_tensorboard(x, x_recon, 'train')
        
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['video']
        recon_loss, x_recon, vq_output = self.forward(x)
        self.log('val/recon_loss', recon_loss, prog_bar=True)
        self.log('val/perplexity', vq_output['perplexity'], prog_bar=True)
        self.log('val/commitment_loss', vq_output['commitment_loss'], prog_bar=True)
        
        # Log images every validation step
        self._log_images_tensorboard(x, x_recon, 'val')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))

    def _log_images_tensorboard(self, original, reconstructed, prefix):
        """Log images directly to TensorBoard using add_figure."""
        try:
            # Workaround for PIL Image.ANTIALIAS issue
            import PIL.Image
            if not hasattr(PIL.Image, 'ANTIALIAS'):
                PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
            
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Take the first sample from the batch
            orig = original[0]  # [C, T, H, W]
            recon = reconstructed[0]  # [C, T, H, W]
            
            # Convert from [-0.5, 0.5] to [0, 1] range
            orig = torch.clamp(orig + 0.5, 0, 1)
            recon = torch.clamp(recon + 0.5, 0, 1)
            
            # Select a few frames from the sequence
            T = orig.shape[1]
            frame_indices = torch.linspace(0, T-1, min(4, T), dtype=torch.long)
            
            # Create comparison figure
            fig, axes = plt.subplots(2, len(frame_indices), figsize=(4 * len(frame_indices), 8))
            if len(frame_indices) == 1:
                axes = axes.reshape(2, 1)
            
            for i, frame_idx in enumerate(frame_indices):
                orig_frame = orig[:, frame_idx]  # [C, H, W]
                recon_frame = recon[:, frame_idx]  # [C, H, W]
                
                # Convert to numpy
                orig_np = orig_frame.detach().cpu().numpy()
                recon_np = recon_frame.detach().cpu().numpy()
                
                # Plot original frame
                if orig_np.shape[0] == 3:  # RGB
                    axes[0, i].imshow(orig_np.transpose(1, 2, 0))
                else:  # Grayscale
                    axes[0, i].imshow(orig_np[0], cmap='gray')
                axes[0, i].set_title(f'Original Frame {frame_idx}')
                axes[0, i].axis('off')
                
                # Plot reconstructed frame
                if recon_np.shape[0] == 3:  # RGB
                    axes[1, i].imshow(recon_np.transpose(1, 2, 0))
                else:  # Grayscale
                    axes[1, i].imshow(recon_np[0], cmap='gray')
                axes[1, i].set_title(f'Reconstructed Frame {frame_idx}')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            
            # Log the figure directly to TensorBoard
            self.logger.experiment.add_figure(f'{prefix}/video_comparison', fig, self.global_step)
            plt.close(fig)  # Close to free memory
            
        except Exception as e:
            # If image logging fails, just skip it to avoid breaking training
            if self.global_step % 100 == 0:
                print(f"Warning: Failed to log images to TensorBoard: {e}")
            pass

    def _test_add_images(self, original, reconstructed, prefix):
        """Test method to try add_images for TensorBoard logging."""
        try:
            # Take the first sample from the batch
            orig = original[0]  # [C, T, H, W]
            recon = reconstructed[0]  # [C, T, H, W]
            
            # Convert from [-0.5, 0.5] to [0, 1] range
            orig = torch.clamp(orig + 0.5, 0, 1)
            recon = torch.clamp(recon + 0.5, 0, 1)
            
            # Select a few frames from the sequence
            T = orig.shape[1]
            frame_indices = torch.linspace(0, T-1, min(4, T), dtype=torch.long)
            
            # Create a batch of images for add_images
            orig_frames = orig[:, frame_indices]  # [C, 4, H, W]
            recon_frames = recon[:, frame_indices]  # [C, 4, H, W]
            
            # Reshape to [N, C, H, W] format for add_images
            orig_batch = orig_frames.permute(1, 0, 2, 3)  # [4, C, H, W]
            recon_batch = recon_frames.permute(1, 0, 2, 3)  # [4, C, H, W]
            
            # Try add_images with different data formats
            print(f"Testing add_images for {prefix} at step {self.global_step}")
            print(f"Original shape: {orig_batch.shape}, Reconstructed shape: {recon_batch.shape}")
            
            # Test 1: Try add_images with NCHW format
            try:
                self.logger.experiment.add_images(f'{prefix}/original_frames', 
                                                orig_batch, 
                                                self.global_step, 
                                                dataformats='NCHW')
                print("✓ add_images with NCHW format worked!")
            except Exception as e:
                print(f"✗ add_images with NCHW failed: {e}")
            
            # Test 2: Try add_images with NHWC format
            try:
                orig_nhwc = orig_batch.permute(0, 2, 3, 1)  # [N, H, W, C]
                recon_nhwc = recon_batch.permute(0, 2, 3, 1)  # [N, H, W, C]
                
                self.logger.experiment.add_images(f'{prefix}/original_frames_nhwc', 
                                                orig_nhwc, 
                                                self.global_step, 
                                                dataformats='NHWC')
                print("✓ add_images with NHWC format worked!")
            except Exception as e:
                print(f"✗ add_images with NHWC failed: {e}")
            
            # Test 3: Try add_image for single frames
            try:
                for i in range(min(2, orig_batch.shape[0])):
                    self.logger.experiment.add_image(f'{prefix}/single_orig_{i}', 
                                                   orig_batch[i], 
                                                   self.global_step, 
                                                   dataformats='CHW')
                print("✓ add_image for single frames worked!")
            except Exception as e:
                print(f"✗ add_image for single frames failed: {e}")
                
        except Exception as e:
            print(f"✗ _test_add_images failed completely: {e}")

    def _save_images(self, original, reconstructed, prefix):
        """Save original and reconstructed video frames to disk."""
        try:
            import os
            import matplotlib.pyplot as plt
            
            # Create output directory
            output_dir = f"training_images/{prefix}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Take the first sample from the batch
            orig = original[0]  # [C, T, H, W]
            recon = reconstructed[0]  # [C, T, H, W]
            
            # Convert from [-0.5, 0.5] to [0, 1] range
            orig = torch.clamp(orig + 0.5, 0, 1)
            recon = torch.clamp(recon + 0.5, 0, 1)
            
            # Select a few frames from the sequence
            T = orig.shape[1]
            frame_indices = torch.linspace(0, T-1, min(4, T), dtype=torch.long)
            
            # Create comparison figure
            fig, axes = plt.subplots(2, len(frame_indices), figsize=(4 * len(frame_indices), 8))
            if len(frame_indices) == 1:
                axes = axes.reshape(2, 1)
            
            for i, frame_idx in enumerate(frame_indices):
                orig_frame = orig[:, frame_idx]  # [C, H, W]
                recon_frame = recon[:, frame_idx]  # [C, H, W]
                
                # Convert to numpy
                orig_np = orig_frame.detach().cpu().numpy()
                recon_np = recon_frame.detach().cpu().numpy()
                
                # Plot original frame
                if orig_np.shape[0] == 3:  # RGB
                    axes[0, i].imshow(orig_np.transpose(1, 2, 0))
                else:  # Grayscale
                    axes[0, i].imshow(orig_np[0], cmap='gray')
                axes[0, i].set_title(f'Original Frame {frame_idx}')
                axes[0, i].axis('off')
                
                # Plot reconstructed frame
                if recon_np.shape[0] == 3:  # RGB
                    axes[1, i].imshow(recon_np.transpose(1, 2, 0))
                else:  # Grayscale
                    axes[1, i].imshow(recon_np[0], cmap='gray')
                axes[1, i].set_title(f'Reconstructed Frame {frame_idx}')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            
            # Save the figure
            filename = f"{output_dir}/step_{self.global_step:06d}.png"
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            plt.close(fig)  # Close to free memory
            
            # Log the file path to TensorBoard as text
            self.logger.experiment.add_text(f'{prefix}/image_path', filename, self.global_step)
            
        except Exception as e:
            # If image saving fails, just skip it to avoid breaking training
            if self.global_step % 100 == 0:
                print(f"Warning: Failed to save images: {e}")
            pass

    def _log_images(self, original, reconstructed, prefix):
        """Log original and reconstructed video frames to TensorBoard."""
        try:
            # Take the first sample from the batch
            orig = original[0]  # [C, T, H, W]
            recon = reconstructed[0]  # [C, T, H, W]
            
            # Convert from [-0.5, 0.5] to [0, 1] range
            orig = torch.clamp(orig + 0.5, 0, 1)
            recon = torch.clamp(recon + 0.5, 0, 1)
            
            # Select a few frames from the sequence (e.g., every 4th frame)
            T = orig.shape[1]
            frame_indices = torch.linspace(0, T-1, min(4, T), dtype=torch.long)
            
            # Log individual frames using add_figure to avoid PIL issues
            for i, frame_idx in enumerate(frame_indices):
                orig_frame = orig[:, frame_idx]  # [C, H, W]
                recon_frame = recon[:, frame_idx]  # [C, H, W]
                
                # Convert to numpy and ensure proper range
                orig_np = orig_frame.detach().cpu().numpy()
                recon_np = recon_frame.detach().cpu().numpy()
                
                # Ensure values are in [0, 1] range
                orig_np = np.clip(orig_np, 0, 1)
                recon_np = np.clip(recon_np, 0, 1)
                
                # Log using add_figure with matplotlib to avoid PIL issues
                import matplotlib.pyplot as plt
                
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                
                # Plot original frame
                if orig_np.shape[0] == 3:  # RGB
                    axes[0].imshow(orig_np.transpose(1, 2, 0))
                else:  # Grayscale
                    axes[0].imshow(orig_np[0], cmap='gray')
                axes[0].set_title(f'Original Frame {i}')
                axes[0].axis('off')
                
                # Plot reconstructed frame
                if recon_np.shape[0] == 3:  # RGB
                    axes[1].imshow(recon_np.transpose(1, 2, 0))
                else:  # Grayscale
                    axes[1].imshow(recon_np[0], cmap='gray')
                axes[1].set_title(f'Reconstructed Frame {i}')
                axes[1].axis('off')
                
                plt.tight_layout()
                
                # Log the figure
                self.logger.experiment.add_figure(f'{prefix}/frame_comparison_{i}', fig, self.global_step)
                plt.close(fig)  # Close to free memory
                
        except Exception as e:
            # If image logging fails, just skip it to avoid breaking training
            # Only print warning occasionally to avoid spam
            if self.global_step % 100 == 0:
                print(f"Warning: Failed to log images: {e}")
            pass

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--embedding_dim', type=int, default=256)
        parser.add_argument('--n_codes', type=int, default=2048)
        parser.add_argument('--n_hiddens', type=int, default=240)
        parser.add_argument('--n_res_layers', type=int, default=4)
        parser.add_argument('--downsample', nargs='+', type=int, default=(4, 4, 4))
        return parser


class AxialBlock(nn.Module):
    def __init__(self, n_hiddens, n_head):
        super().__init__()
        kwargs = dict(shape=(0,) * 3, dim_q=n_hiddens,
                      dim_kv=n_hiddens, n_head=n_head,
                      n_layer=1, causal=False, attn_type='axial')
        self.attn_w = MultiHeadAttention(attn_kwargs=dict(axial_dim=-2),
                                         **kwargs)
        self.attn_h = MultiHeadAttention(attn_kwargs=dict(axial_dim=-3),
                                         **kwargs)
        self.attn_t = MultiHeadAttention(attn_kwargs=dict(axial_dim=-4),
                                         **kwargs)

    def forward(self, x):
        x = shift_dim(x, 1, -1)
        x = self.attn_w(x, x, x) + self.attn_h(x, x, x) + self.attn_t(x, x, x)
        x = shift_dim(x, -1, 1)
        return x


class AttentionResidualBlock(nn.Module):
    def __init__(self, n_hiddens):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            SamePadConv3d(n_hiddens, n_hiddens // 2, 3, bias=False),
            nn.BatchNorm3d(n_hiddens // 2),
            nn.ReLU(),
            SamePadConv3d(n_hiddens // 2, n_hiddens, 1, bias=False),
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            AxialBlock(n_hiddens, 2)
        )

    def forward(self, x):
        return x + self.block(x)

class Codebook(nn.Module):
    def __init__(self, n_codes, embedding_dim):
        super().__init__()
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        # z: [b, c, t, h, w]
        self._need_init = False
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        y = self._tile(flat_inputs)

        d = y.shape[0]
        _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))

    def forward(self, z):
        # z: [b, c, t, h, w]
        if self._need_init and self.training:
            self._init_embeddings(z)
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * flat_inputs @ self.embeddings.t() \
                    + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True)

        encoding_indices = torch.argmin(distances, dim=1)
        encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(flat_inputs)
        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:])

        embeddings = F.embedding(encoding_indices, self.embeddings)
        embeddings = shift_dim(embeddings, -1, 1)

        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

        # EMA codebook update
        if self.training:
            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            usage = (self.N.view(self.n_codes, 1) >= 1).float()
            self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        embeddings_st = (embeddings - z).detach() + z

        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return dict(embeddings=embeddings_st, encodings=encoding_indices,
                    commitment_loss=commitment_loss, perplexity=perplexity)

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings

class Encoder(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, downsample):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.convs = nn.ModuleList()
        max_ds = n_times_downsample.max()
        for i in range(max_ds):
            in_channels = 3 if i == 0 else n_hiddens
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            conv = SamePadConv3d(in_channels, n_hiddens, 4, stride=stride)
            self.convs.append(conv)
            n_times_downsample -= 1
        self.conv_last = SamePadConv3d(in_channels, n_hiddens, kernel_size=3)

        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hiddens)
              for _ in range(n_res_layers)],
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU()
        )

    def forward(self, x):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h))
        h = self.conv_last(h)
        h = self.res_stack(h)
        return h


class Decoder(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, upsample):
        super().__init__()
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hiddens)
              for _ in range(n_res_layers)],
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU()
        )

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        self.convts = nn.ModuleList()
        for i in range(max_us):
            out_channels = 3 if i == max_us - 1 else n_hiddens
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            convt = SamePadConvTranspose3d(n_hiddens, out_channels, 4,
                                           stride=us)
            self.convts.append(convt)
            n_times_upsample -= 1

    def forward(self, x):
        h = self.res_stack(x)
        for i, convt in enumerate(self.convts):
            h = convt(h)
            if i < len(self.convts) - 1:
                h = F.relu(h)
        return h


# Does not support dilation
class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input))


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input))

