import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from videogpt import VideoGPT, VideoData


def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VideoGPT.add_model_specific_args(parser)
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset (HDF5 file or directory)')
    parser.add_argument('--dataset_type', type=str, default='auto',
                       choices=['auto', 'moving_mnist'],
                       help='Type of dataset to use')
    parser.add_argument('--resolution', type=int, default=128,
                       help='Target resolution for videos')
    parser.add_argument('--sequence_length', type=int, default=16,
                       help='Number of frames per video sequence')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--num_digits', type=int, default=2, choices=[1, 2],
                       help='Number of digits per video (for Moving MNIST)')
    parser.add_argument('--videos_per_digit', type=int, default=1000,
                       help='Number of videos per digit class (for Moving MNIST)')
    args = parser.parse_args()
    
    # Auto-detect dataset type if not specified
    if args.dataset_type == 'auto':
        if args.data_path == 'moving_mnist' or 'moving_mnist' in args.data_path.lower():
            args.dataset_type = 'moving_mnist'
        else:
            args.dataset_type = 'auto'  # Will use default VideoDataset/HDF5Dataset logic

    data = VideoData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.test_dataloader()

    args.class_cond_dim = data.n_classes if args.class_cond else None
    model = VideoGPT(args)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/loss', mode='min', save_top_k=-1))

    kwargs = dict()
    if args.gpus > 1:
        # find_unused_parameters = False to support gradient checkpointing
        kwargs = dict(distributed_backend='ddp', gpus=args.gpus,
                      plugins=[pl.plugins.DDPPlugin(find_unused_parameters=False)])
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks,
                                            max_steps=args.max_steps, **kwargs)

    trainer.fit(model, data)


if __name__ == '__main__':
    main()

