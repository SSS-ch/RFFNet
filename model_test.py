import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.RFF import RFF
from utils.data_loader import DatasetFromFolder
from utils.metric import calculate_psnr_3d, ssim_3d


def main(opt):
    hr_path = opt.hr_path
    pre_save_path = opt.pre_save_path
    pth_path = opt.pth_path
    upscale_factor = opt.upscale_factor

    model = RFF(1, upscale_factor)

    if torch.cuda.is_available():
        model.cuda()
    else:
        print("CPU")

    checkpoint = torch.load(pth_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    if not os.path.exists(pre_save_path):
        os.makedirs(pre_save_path)

    test_set = DatasetFromFolder(hr_path, upscale_factor)

    files = sorted([f for f in os.listdir(hr_path) if f.endswith('.npy')])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False)

    results = {'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    start = time.time()

    for i, (hr, lr, m) in enumerate(tqdm(test_loader)):
        batch_size = hr.size(0)
        results['batch_sizes'] += batch_size

        if torch.cuda.is_available():
            hr = hr.cuda()
            lr = lr.cuda()
            m = m.cuda()
            m = m[0]

        with torch.no_grad():
            predict = model(lr)

        predict = predict * m

        batch_psnr = calculate_psnr_3d(hr * m, predict)
        results['psnr'] += batch_psnr * batch_size

        batch_ssim = ssim_3d(hr * m, predict).item()
        results['ssims'] += batch_ssim * batch_size
        results['ssim'] = results['ssims'] / results['batch_sizes']

        np.save(os.path.join(pre_save_path, files[i]), predict.squeeze().cpu().numpy())

    end = time.time()
    print(f"Average PSNR: {results['psnr'] / results['batch_sizes']:.4f} dB")
    print(f"Average SSIM: {results['ssim']:.4f}")
    print(f"Total time: {(end - start):.2f} s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test RFFNet for 3D MRI Super-Resolution')

    parser.add_argument('--hr_path', type=str, required=True,
                        help='Path to high-resolution test data (.npy)')

    parser.add_argument('--pre_save_path', type=str, required=True,
                        help='Directory to save reconstructed SR volumes')

    parser.add_argument('--pth_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth)')

    parser.add_argument('--upscale_factor', type=int, default=2, choices=[2, 4, 8],
                        help='Super-resolution upscale factor')

    opt = parser.parse_args()
    main(opt)
