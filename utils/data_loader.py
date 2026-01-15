import os

import torch

from scipy import ndimage

from torch.utils.data import Dataset, DataLoader

import numpy as np

import torch.nn.functional as F
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.npy', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class testDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(testDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):

        hr_image = np.array(np.load(self.image_filenames[index]),dtype=np.float32)
        hx,hy,hz=hr_image.shape[0],hr_image.shape[1],hr_image.shape[2]
        lx,ly,lz=hx//self.upscale_factor,hy//self.upscale_factor,hz//self.upscale_factor
        datamax=hr_image.max()
        #datamax=2408321.5
        #print(datamax)
        hr_image=hr_image/datamax
        hr = hr_image.reshape(1, 1, hx, hy, hz)
        hr=torch.from_numpy(hr)
        blurred_image = ndimage.gaussian_filter(hr_image, sigma=1)
        hr_data=blurred_image.reshape(1,1, hx, hy, hz)
        input_tensor=torch.from_numpy(hr_data)
        output_tensor = F.interpolate(input_tensor, size=(lx, ly, lz), mode='trilinear', align_corners=False)
        pre_tensor= F.interpolate(output_tensor, size=(hx, hy, hz), mode='trilinear', align_corners=False)
        return hr.squeeze(dim=0),output_tensor.squeeze(dim=0),pre_tensor.squeeze(dim=0),datamax
    def __len__(self):
        return len(self.image_filenames)




class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(DatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]
        a=len(self.filenames)
        print(f"data len:{a}")
    def __getitem__(self, index):

        hr_data = np.array(np.load(self.filenames[index]),dtype=np.float32)
        hx,hy,hz=hr_data.shape[0],hr_data.shape[1],hr_data.shape[2]
        lx,ly,lz=hx//self.upscale_factor,hy//self.upscale_factor,hz//self.upscale_factor
        datamax=2408321.5   #Kirby21
        #datamax = 32767.0   #BraTs2019
        #datamax=3868   #IXI-T2
        #datamax=6230.0 #IXI-PD
        #datamax=2408321.5  #Kirby21
        #datamax=3868.0
        #datamax=13160.0   # IXI-T1
        hr_data=hr_data/datamax
        hr = hr_data.reshape(1, 1, hx, hy, hz)
        hr=torch.from_numpy(hr)
        #print(f"hrmax:{hr_image.max()},min:{hr_image.min()}")
        blurred_image = ndimage.gaussian_filter(hr_data, sigma=1)

        bl_data=blurred_image.reshape(1,1, hx, hy, hz)

        input_tensor=torch.from_numpy(bl_data)

        lr = F.interpolate(input_tensor, size=(lx, ly, lz), mode='trilinear', align_corners=False)

        return hr.squeeze(dim=0),lr.squeeze(dim=0),datamax

    def __len__(self):
        return len(self.filenames)








