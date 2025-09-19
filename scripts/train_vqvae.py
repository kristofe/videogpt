import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from videogpt import VQVAE, VideoData
from videogpt.utils import get_device, get_device_count, is_distributed_available


def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    # Add basic trainer arguments manually since add_argparse_args was removed in PL 2.x
    parser.add_argument('--gpus', type=int, default=None, help='Number of GPUs to use')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum number of epochs')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum number of steps')
    parser = VQVAE.add_model_specific_args(parser)
    parser.add_argument('--data_path', type=str, default='/home/wilson/data/datasets/bair.hdf5')
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use: cuda, mps, cpu, or auto (default: auto)')
    args = parser.parse_args()

    # Get the appropriate device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Update device-specific arguments
    if device.type == 'cuda':
        if not hasattr(args, 'gpus') or args.gpus is None:
            args.gpus = get_device_count()
    else:
        # For MPS and CPU, set gpus to 0 to use CPU/MPS
        args.gpus = 0

    data = VideoData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.test_dataloader()
    model = VQVAE(args)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss', mode='min'))

    kwargs = dict()
    if args.gpus > 1 and is_distributed_available():
        kwargs = dict(distributed_backend='ddp', gpus=args.gpus)
    elif device.type == 'mps':
        # MPS backend for PyTorch Lightning
        kwargs = dict(accelerator='mps', devices=1)
    elif device.type == 'cpu':
        kwargs = dict(accelerator='cpu', devices=1)
    
    # Create trainer with appropriate arguments
    trainer_kwargs = {
        'callbacks': callbacks,
        'max_epochs': args.max_epochs,
        'max_steps': 200000,
        **kwargs
    }
    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(model, data)


if __name__ == '__main__':
    main()

