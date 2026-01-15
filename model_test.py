import os
import time

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F
import nibabel as nib

from utils.data_loader import DatasetFromFolder
from utils.metric import calculate_psnr_3d, ssim_3d
from model.RFF import RFF


def save_nii(npy_file_path,nii_file_path):
    npy_data = np.load(npy_file_path)

    npy_data = np.flipud(npy_data)
    #npy_data = np.flip(npy_data, axis=1)

    affine = np.eye(4)
    nifti_image = nib.Nifti1Image(npy_data, affine)

    nib.save(nifti_image, nii_file_path)
    print(f"Saved NIfTI image to {nii_file_path}")


def bic_test(hr_path,pre_save_path):
    if not os.path.exists(pre_save_path):
        os.makedirs(pre_save_path)
    test_set = DatasetFromFolder(hr_path, 2)
    namelist = []
    files1 = sorted([os.path.join(hr_path, f) for f in os.listdir(hr_path) if f.endswith('.npy')])
    for npy_file in files1:
        # name = npy_file.split('/')[-1]
        name = npy_file.split('/')[-1]
        namelist.append(name)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, drop_last=False)
    test_bar = tqdm(test_loader)
    results = {'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    i = 0
    start = time.time()
    for hr, lr, m in test_bar:
        batch_size = hr.size(0)
        results['batch_sizes'] += batch_size
        hx, hy, hz = hr.shape[2], hr.shape[3], hr.shape[4]
        #print(f"{hx},{hy},{hz}")
        predict=F.interpolate(lr, size=(hx, hy, hz), mode='trilinear', align_corners=False)
        #predict = predict * m
        predict=predict.float()
        hr=hr.cuda()
        lr=lr.cuda()
        m=m.cuda()
        m=m[0]

        predict=predict.cuda()
        predict=predict*m
        batch_psnr = calculate_psnr_3d(hr * m, predict)
        results['psnr'] += batch_psnr * batch_size

        batch_ssim = ssim_3d(hr * m, predict).item()
        results['ssims'] += batch_ssim * batch_size
        results['ssim'] = results['ssims'] / results['batch_sizes']
        test_bar.set_description(
            desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                results['psnr'] / results['batch_sizes'], results['ssim']))

        predict_np = predict.squeeze().cpu().numpy()
        np.save(os.path.join(pre_save_path, namelist[i]), predict_np)
    end = time.time()
    print(f"Batch time: {(end - start) * 1000:.2f} ms")

def test(hr_path,pre_save_path,pth_path):
    hr_path = hr_path
    pre_save_path = pre_save_path
    pth_path = pth_path

    model=RFF(1,2)
    if torch.cuda.is_available():
        model.cuda()
    else:
        print("CPU")
    pth_path = torch.load(pth_path)

    if not os.path.exists(pre_save_path):
        os.makedirs(pre_save_path)

    model.load_state_dict(pth_path)
    test_set = DatasetFromFolder(hr_path,2)
    namelist = []
    files1 = sorted([os.path.join(hr_path, f) for f in os.listdir(hr_path) if f.endswith('.npy')])
    for npy_file in files1:
        # name = npy_file.split('/')[-1]
        name = npy_file.split('/')[-1]
        #print(name)
        namelist.append(name)

    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, drop_last=False)
    test_bar = tqdm(test_loader)
    results = {'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    start = time.time()
    i=0
    for hr, lr,m in test_bar:

        batch_size = hr.size(0)
        results['batch_sizes'] += batch_size
        if torch.cuda.is_available():
            hr = hr.cuda()
            lr = lr.cuda()
            m=m.cuda()
            m=m[0]
        with torch.no_grad():
            predict = model(lr)

        predict=predict*m
        batch_psnr = calculate_psnr_3d(hr*m, predict)
        results['psnr'] += batch_psnr * batch_size

        batch_ssim = ssim_3d(hr*m, predict).item()
        results['ssims'] += batch_ssim * batch_size
        results['ssim'] = results['ssims'] / results['batch_sizes']
        test_bar.set_description(
            desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                results['psnr'] / results['batch_sizes'], results['ssim']))
        predict_np = predict.squeeze().cpu().numpy()
        np.save(os.path.join(pre_save_path, namelist[i]), predict_np)
        i = i + 1
    end = time.time()
    print(f"Batch time: {(end - start) * 1000:.2f} ms")
