from torch.utils.data import Dataset
import os
from torchvision.io import read_image
from matplotlib.image import imread

class ASLDataset(Dataset):
    
    def __init__(self, set_type, transform=None, target_transform=None, size_per_class=3000):

        self.transform = transform
        self.target_transform = target_transform
        
        self.paths  = []
        self.categories = []
        
        classes = [chr(x) for x in range(ord('A'),ord('Z')+1)] + ['del', 'nothing','space']

        start_range = 0
        end_range = 0
        if set_type == 'train':
            end_range = int(size_per_class * 0.85)
        elif set_type == 'test':
            start_range = int(size_per_class * 0.85)
            end_range = size_per_class + 1
            
        for idx, cat in enumerate(classes):
            self.paths.extend([f'{cat}/{cat}{n}.jpg' for n in range(start_range, end_range)])
            self.categories.extend([idx for j in range(end_range-start_range)])
            
            
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        
        img_path = './asl_alphabet_train/asl_alphabet_train'
        
        img_path = os.path.join(img_path, self.paths[idx])
        image = imread(img_path)
        label = self.categories[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
