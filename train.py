import argparse
import datetime
import os

from torch import nn
import pandas as pd
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metric import calculate_psnr_3d, ssim_3d, loss_plot

from model.RFF import RFF
from utils.data_loader import DatasetFromFolder

parser = argparse.ArgumentParser(description='Train Super Resolution Models')

parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=200, type=int, help='train epoch number')
parser.add_argument('--batch_size', default=32, type=int, help='train epoch number')

if __name__ == '__main__':
    opt = parser.parse_args([])
    UPSCALE_FACTOR = opt.upscale_factor
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs
    current_time = datetime.datetime.now()
    current_time_str = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    out_path = os.path.join('logs', f"{current_time_str}")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    train_hr_path = "dataset/kirby21/train/hr_patch"
    val_hr_path = "dataset/kirby21/val/hr_patch"
    train_set = DatasetFromFolder(train_hr_path, UPSCALE_FACTOR)
    val_set = DatasetFromFolder(val_hr_path, UPSCALE_FACTOR)

    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = RFF(1, UPSCALE_FACTOR)
    # print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    # 参数量
    if torch.cuda.is_available():
        model.cuda()
    else:
        print("CPU")
    initial_lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    content_criterion = nn.MSELoss()
    results = {'train_loss': [], 'val_loss': [], 'psnr': [], 'ssim': []}
    save_loss = 1.0
    savepsnr = 1.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'train_loss': 0}

        model.train()
        for hr, lr, _ in train_bar:
            optimizer.zero_grad()
            current_lr = optimizer.param_groups[0]['lr']
            batch_size = hr.size(0)  # batch_size=batch_size
            running_results['batch_sizes'] += batch_size
            hr, lr = hr.cuda(), lr.cuda()

            predict_hr = model(lr)

            content_loss = content_criterion(hr, predict_hr).cuda()
            content_loss.backward()
            optimizer.step()
            running_results['train_loss'] += content_loss.item() * batch_size
            train_bar.set_description(desc='Train: [%d/%d] learning_rate:%.6f Loss_G: %.4f' % (
                epoch, NUM_EPOCHS, current_lr,
                running_results['train_loss'] / running_results['batch_sizes']))

        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'val_loss': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_hr, val_lr, m in val_bar:
                batch_size = val_hr.size(0)
                valing_results['batch_sizes'] += batch_size

                if torch.cuda.is_available():
                    val_hr = val_hr.cuda()
                    val_lr = val_lr.cuda()
                    m = m.cuda()
                    m = m[0]
                val_predict_hr = model(val_lr)
                content_loss = content_criterion(val_hr, val_predict_hr).cuda()
                val_predict_hr = val_predict_hr * m

                batch_psnr = calculate_psnr_3d(val_hr * m, val_predict_hr)
                valing_results['psnr'] += batch_psnr * batch_size
                valing_results['val_loss'] += content_loss.item() * batch_size

                batch_ssim = ssim_3d(val_hr * m, val_predict_hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='Val: [converting LR images to SR images] val_loss: %.4f PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['val_loss'] / valing_results['batch_sizes'],
                        valing_results['psnr'] / valing_results['batch_sizes'], valing_results['ssim']))
        if epoch % 20 == 0 and epoch != 0:
            torch.save(model.state_dict(), os.path.join(out_path, 'model_epoch_%d.pth' % (epoch)))
        if valing_results['val_loss'] / valing_results['batch_sizes'] < save_loss:
            torch.save(model.state_dict(), os.path.join(out_path, '%d_best_model.pth' % (epoch)))
            save_loss = valing_results['val_loss'] / valing_results['batch_sizes']

        if valing_results['psnr'] / valing_results['batch_sizes'] > savepsnr:
            torch.save(model.state_dict(), os.path.join(out_path, '%d_best_psnr.pth' % (epoch)))
            savepsnr = valing_results['psnr'] / valing_results['batch_sizes']

        results['train_loss'].append(running_results['train_loss'] / running_results['batch_sizes'])
        results['val_loss'].append(valing_results['val_loss'] / valing_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'] / valing_results['batch_sizes'])

        results['ssim'].append(valing_results['ssim'])

        if epoch % 2 == 0 and epoch != 0:
            data_frame = pd.DataFrame(
                data={'Train_Loss': results['train_loss'], 'Val_Loss': results['val_loss'], 'PSNR': results['psnr'],
                      'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(os.path.join(out_path, "train_results.csv"), index_label='Epoch')

        loss_plot(results['train_loss'], results['val_loss'], out_path)
    end_time = datetime.datetime.now()
    end_time_str = end_time.strftime("%Y_%m_%d_%H_%M_%S")
    print(f"end_time:{end_time_str}")

