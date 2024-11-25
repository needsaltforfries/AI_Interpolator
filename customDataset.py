import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms

class ImgTripletDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return int((len(self.annotations)-1701)) # 51313
    
    def __getitem__(self, index):
        #make path for folders
        #file structure is like xxxxx/yyyy
        img_set = str(self.annotations.iloc[index, 0]).zfill(5)
        img_subdir = str(self.annotations.iloc[index, 1]).zfill(4)
        
        #make image paths
        base_path = os.path.join(self.root_dir, img_set, img_subdir)
        img1_path = os.path.join(base_path, 'im1.png')
        img2_path = os.path.join(base_path, 'im2.png')
        img3_path = os.path.join(base_path, 'im3.png')
        
        #get image
        im1 = read_image(img1_path)
        im2 = read_image(img2_path)
        im3 = read_image(img3_path)
        #lables are folder directory
        label = self.annotations.iloc[index, 0]
        label2 = self.annotations.iloc[index, 1]
        #transform if necessary
        if self.transform:
            to_pil = transforms.ToPILImage()
            im1 = self.transform(to_pil(im1))
            im2 = self.transform(to_pil(im2))
            im3 = self.transform(to_pil(im3))
        
        return (im1, im2, im3), (label, label2)