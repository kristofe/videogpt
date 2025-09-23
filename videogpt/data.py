import os
import os.path as osp
import math
import random
import pickle
import warnings

import glob
import h5py
import numpy as np

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.datasets.video_utils import VideoClips
import pytorch_lightning as pl


class VideoDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm']

    def __init__(self, data_folder, sequence_length, train=True, resolution=64):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution

        folder = osp.join(data_folder, 'train' if train else 'test')
        files = sum([glob.glob(osp.join(folder, '**', f'*.{ext}'), recursive=True)
                     for ext in self.exts], [])

        # hacky way to compute # of classes (count # of unique parent directories)
        self.classes = list(set([get_parent_dir(f) for f in files]))
        self.classes.sort()
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}

        warnings.filterwarnings('ignore')
        cache_file = osp.join(folder, f"metadata_{sequence_length}.pkl")
        if not osp.exists(cache_file):
            clips = VideoClips(files, sequence_length, num_workers=32)
            pickle.dump(clips.metadata, open(cache_file, 'wb'))
        else:
            metadata = pickle.load(open(cache_file, 'rb'))
            clips = VideoClips(files, sequence_length,
                               _precomputed_metadata=metadata)
        self._clips = clips

    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return self._clips.num_clips()

    def __getitem__(self, idx):
        resolution = self.resolution
        video, _, _, idx = self._clips.get_clip(idx)

        class_name = get_parent_dir(self._clips.video_paths[idx])
        label = self.class_to_label[class_name]
        return dict(video=preprocess(video, resolution), label=label)


def get_parent_dir(path):
    return osp.basename(osp.dirname(path))


def preprocess(video, resolution, sequence_length=None):
    # video: THWC, {0, ..., 255}
    video = video.permute(0, 3, 1, 2).float() / 255. # TCHW
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous() # CTHW

    video -= 0.5

    return video


class HDF5Dataset(data.Dataset):
    """ Generic dataset for data stored in h5py as uint8 numpy arrays.
    Reads videos in {0, ..., 255} and returns in range [-0.5, 0.5] """
    def __init__(self, data_file, sequence_length, train=True, resolution=64):
        """
        Args:
            data_file: path to the pickled data file with the
                following format:
                {
                    'train_data': [B, H, W, 3] np.uint8,
                    'train_idx': [B], np.int64 (start indexes for each video)
                    'test_data': [B', H, W, 3] np.uint8,
                    'test_idx': [B'], np.int64
                }
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution

        # read in data
        self.data_file = data_file
        self.data = h5py.File(data_file, 'r')
        self.prefix = 'train' if train else 'test'
        self._images = self.data[f'{self.prefix}_data']
        self._idx = self.data[f'{self.prefix}_idx']
        self.size = len(self._idx)

    @property
    def n_classes(self):
        raise Exception('class conditioning not support for HDF5Dataset')

    def __getstate__(self):
        state = self.__dict__
        state['data'] = None
        state['_images'] = None
        state['_idx'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.data = h5py.File(self.data_file, 'r')
        self._images = self.data[f'{self.prefix}_data']
        self._idx = self.data[f'{self.prefix}_idx']

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        start = self._idx[idx]
        end = self._idx[idx + 1] if idx < len(self._idx) - 1 else len(self._images)
        assert end - start >= 0

        # Check if we have enough frames for the sequence length
        available_frames = end - start
        if available_frames < self.sequence_length:
            # If not enough frames, pad with the last frame or repeat frames
            if available_frames == 0:
                # Edge case: no frames available, create a zero tensor
                video = torch.zeros(self.sequence_length, *self._images.shape[1:])
            else:
                # Repeat the available frames to reach sequence_length
                video_frames = self._images[start:end]
                repeat_times = (self.sequence_length + available_frames - 1) // available_frames
                video_frames = np.tile(video_frames, (repeat_times, 1, 1, 1))
                video = torch.tensor(video_frames[:self.sequence_length])
        else:
            # Normal case: randomly sample a subsequence
            max_start = end - start - self.sequence_length
            if max_start > 0:
                start_offset = np.random.randint(low=0, high=max_start)
                start = start + start_offset
            else:
                # Edge case: exactly sequence_length frames available
                start = start
            assert start < start + self.sequence_length <= end
            video = torch.tensor(self._images[start:start + self.sequence_length])
        
        return dict(video=preprocess(video, self.resolution))


class MovingMNISTDataset(data.Dataset):
    """
    Moving MNIST Dataset that integrates with VideoGPT.
    This is a simplified wrapper around the full MovingMNISTDataset class.
    """
    
    def __init__(self, data_file, sequence_length, train=True, resolution=64, **kwargs):
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution
        
        # Import here to avoid circular imports
        from .moving_mnist import MovingMNISTDataset as FullMovingMNISTDataset
        
        # Create the full dataset
        self.dataset = FullMovingMNISTDataset(
            sequence_length=sequence_length,
            train=train,
            resolution=resolution,
            **kwargs
        )
    
    @property
    def n_classes(self):
        return self.dataset.n_classes
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class VideoData(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args

    @property
    def n_classes(self):
        dataset = self._dataset(True)
        return dataset.n_classes


    def _dataset(self, train):
        # Check if this is a Moving MNIST dataset request
        if hasattr(self.args, 'dataset_type') and self.args.dataset_type == 'moving_mnist':
            return MovingMNISTDataset(
                data_file=None,  # Not used for Moving MNIST
                sequence_length=self.args.sequence_length,
                train=train,
                resolution=self.args.resolution,
                num_digits=getattr(self.args, 'num_digits', 2),
                videos_per_digit=getattr(self.args, 'videos_per_digit', 1000)
            )
        
        # Default behavior for other datasets
        Dataset = VideoDataset if osp.isdir(self.args.data_path) else HDF5Dataset
        dataset = Dataset(self.args.data_path, self.args.sequence_length,
                          train=train, resolution=self.args.resolution)
        return dataset


    def _dataloader(self, train):
        dataset = self._dataset(train)
        if dist.is_initialized():
            sampler = data.distributed.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
            )
        else:
            sampler = None
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=sampler is None
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader(True)

    def val_dataloader(self):
        return self._dataloader(False)

    def test_dataloader(self):
        return self.val_dataloader()
