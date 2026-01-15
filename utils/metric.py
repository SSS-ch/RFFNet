import os
from math import exp

import torch
import torch.nn.functional as F

from math import log10

from tqdm import tqdm


from torch.utils.data import DataLoader




from matplotlib import pyplot as plt
def loss_plot(train_loss,val_loss,log_dir):
    iters = range(len(train_loss))

    plt.figure()
    plt.plot(iters, train_loss, 'red', linewidth=2, label='train loss')
    plt.plot(iters, val_loss, 'coral', linewidth=2, label='val loss')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")

    plt.savefig(os.path.join(log_dir, "epoch_loss.png"))

    plt.cla()
    plt.close("all")

def calculate_psnr_3d(hr, sr):

    mse = torch.mean((sr - hr) ** 2)
    psnr = 10. * log10(((hr.max())** 2) / mse)
    return psnr

def gaussian_kernel(window_size, sigma):

    kernel = torch.tensor([exp(-(x - window_size//2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    kernel = kernel / kernel.sum()
    kernel_3d = kernel[:, None, None] * kernel[None, :, None] * kernel[None, None, :]
    return kernel_3d

def create_window(window_size, channel):

    kernel = gaussian_kernel(window_size, sigma=1.5).unsqueeze(0).unsqueeze(0)
    window = kernel.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window

def ssim_3d(img1, img2, window_size=11, val_range=None):

    if val_range is None:
        max_val = torch.max(img1)
        min_val = torch.min(img1)
        L = max_val - min_val
    else:
        L = val_range

    padd = window_size // 2
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv3d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv3d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


