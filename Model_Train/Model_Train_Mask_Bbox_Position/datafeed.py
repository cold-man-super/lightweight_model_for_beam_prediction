import pandas as pd
from torch.utils.data import Dataset
from skimage import io
import ast
import torch





def create_samples(root, shuffle=False, nat_sort=False):
    f = pd.read_csv(root)
    data_samples = []
    for idx, row in f.iterrows():
        img_paths = row.values
        data_samples.append(img_paths)
    return data_samples

class DataFeed(Dataset):
    def __init__(self,csv_file, root_dir, nat_sort=False, transform=None, init_shuflle=True):
        self.root = root_dir
        self.samples = create_samples(self.root, shuffle=init_shuflle, nat_sort=nat_sort)
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = io.imread(sample[1])
        bbox_gps = ast.literal_eval(self.data.loc[idx, 'input'])
        label = self.data.loc[idx, 'index']
        label = label-1   #true beam index is from 1 to 32
        if self.transform:
            img = self.transform(img)
        A,B,C = img.shape
        if(A==3):
            img=torch.zeros(1, 32, 32)
        bbox_gps = torch.tensor(bbox_gps, dtype=torch.float32)
        bbox_gps = bbox_gps.unsqueeze(0)
        return (img, bbox_gps, label)