import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms

class ATD12KDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the videos.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.video_folders = sorted(os.listdir(root_dir))
        
    def __len__(self):
        return len(self.video_folders)
    
    def __getitem__(self, index):
        video_folder = self.video_folders[index]
        video_folder_path = os.path.join(self.root_dir, video_folder)
        
        frames = sorted(os.listdir(video_folder_path))
        frame_paths = [os.path.join(video_folder_path, frame) for frame in frames]

        frames_images = [read_image(frame_path) for frame_path in frame_paths]
        if self.transform:
            to_pil = transforms.ToPILImage()
            frames_images = [self.transform(to_pil(img)) for img in frames_images]
        return (frames_images[0], frames_images[1], frames_images[2])
