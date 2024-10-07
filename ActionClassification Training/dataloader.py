from pytorchvideo.data import make_clip_sampler, labeled_video_dataset
from torch.utils.data import DataLoader

from pytorchvideo.transforms import (
    ApplyTransformToKey, 
    RandomShortSideScale,
    UniformTemporalSubsample,
    
)

from torchvision.transforms import (
    functional as F,
    Compose,
    RandomHorizontalFlip,
    Lambda,
)

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
import numbers

def divide_by_255(tensor):
    return tensor / 255.0

class ResizeVideo:
    def __init__(self):
        pass

    def __call__(self, clip):
        return F.resize(clip, (320, 320), interpolation=F.InterpolationMode.BILINEAR)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
    
class train_dataloader(DataLoader):
    def __init__(self, dataset_df, batch_size, num_workers):
        video_transform = Compose([
            ApplyTransformToKey(key = 'video',
            transform = Compose([
                UniformTemporalSubsample(60),
                Lambda(divide_by_255),
                NormalizeVideo(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
                ResizeVideo(),
                RandomHorizontalFlip(p=0.5),
            ])),
        ])
        dataset = labeled_video_dataset(
            dataset_df,
            clip_sampler=make_clip_sampler('random', 2),
            transform=video_transform,
            decode_audio=False,
        )
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        
class test_dataloader(DataLoader):
    def __init__(self, dataset_df, batch_size, num_workers):
        video_transform = Compose([
        ApplyTransformToKey(key = 'video',
            transform = Compose([
                UniformTemporalSubsample(60),
                Lambda(divide_by_255),
                NormalizeVideo(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
                ResizeVideo(),
            ])),
        ])
        dataset = labeled_video_dataset(
            dataset_df,
            clip_sampler=make_clip_sampler('random', 2),
            transform=video_transform,
            decode_audio=False,
        )
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)