"""
Evaluate the performance of different similarity metrics
"""
import sys
import os
import struct
import collections
import glob
import time
from os import path
import cv2 as cv
DIR_PATH = path.dirname(os.path.realpath(__file__))
sys.path.append(path.join(DIR_PATH, ".."))
import numpy as np
import torch
import torch.nn.functional as F
import math
from argparse import ArgumentParser
from math import exp, sqrt


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel, std=1.5):
    _1D_window = gaussian(window_size, std).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def add_noise_func(img_input, noise_type='gaussian', save=True):
    print("Adding noise of", noise_type)
    img = np.copy(img_input)
    height, width, channel = img.shape
    if noise_type == 'gaussian':
        mean = 0.1
        std = 0.1
        noise = np.random.normal(mean, std, img.shape).astype(np.float32)
        noisy_img = img + noise*255.
    elif noise_type == 'slat':
        probability = 0.1
        num_pixels = int(probability * height * width)
        indices = np.random.choice(range(height * width), num_pixels, replace=False)
        img = img.reshape(-1, channel)
        img[indices,:] = 0.3 * 255.
        noisy_img = img.reshape(height, width, channel)
    elif noise_type == 'possion':
        noisy_img = np.random.poisson(img)
    elif noise_type == 'quantization':
        levels = 2 ** 3
        noisy_img = np.round(img * levels) / levels
    else:
        raise NotImplementedError("the noise type has not been implimented!")

    if save:
        cv.imwrite('raw_gray.png', img_input)
        cv.imwrite('noisy_gray.png', noisy_img)
    return noisy_img


def function_ncc(ref_gray, src_grays):
    # ref_gray: [1, batch_size, 121, 1]
    # src_grays: [nsrc, batch_size, 121, 1]
    ref_gray = ref_gray.permute(1, 0, 3, 2)  # [batch_size, 1, 1, 121]
    src_grays = src_grays.permute(1, 0, 3, 2)  # [batch_size, nsrc, 1, 121]

    ref_src = ref_gray * src_grays  # [batch_size, nsrc, 1, npatch]

    bs, nsrc, nc, npatch = src_grays.shape
    patch_size = int(sqrt(npatch))
    ref_gray = ref_gray.view(bs, 1, 1, patch_size, patch_size).view(-1, 1, patch_size, patch_size)
    src_grays = src_grays.view(bs, nsrc, 1, patch_size, patch_size).contiguous().view(-1, 1, patch_size, patch_size)
    ref_src = ref_src.view(bs, nsrc, 1, patch_size, patch_size).contiguous().view(-1, 1, patch_size, patch_size)

    ref_sq = ref_gray.pow(2)
    src_sq = src_grays.pow(2)

    filters = torch.ones(1, 1, patch_size, patch_size, device=ref_gray.device)
    padding = patch_size // 2

    ref_sum = F.conv2d(ref_gray, filters, stride=1, padding=padding)[:, :, padding, padding]
    src_sum = F.conv2d(src_grays, filters, stride=1, padding=padding)[:, :, padding, padding].view(bs, nsrc)
    ref_sq_sum = F.conv2d(ref_sq, filters, stride=1, padding=padding)[:, :, padding, padding]
    src_sq_sum = F.conv2d(src_sq, filters, stride=1, padding=padding)[:, :, padding, padding].view(bs, nsrc)
    ref_src_sum = F.conv2d(ref_src, filters, stride=1, padding=padding)[:, :, padding, padding].view(bs, nsrc)

    u_ref = ref_sum / npatch
    u_src = src_sum / npatch

    cross = ref_src_sum - u_src * ref_sum - u_ref * src_sum + u_ref * u_src * npatch
    ref_var = ref_sq_sum - 2 * u_ref * ref_sum + u_ref * u_ref * npatch
    src_var = src_sq_sum - 2 * u_src * src_sum + u_src * u_src * npatch

    cc = cross * cross / (ref_var * src_var + 1e-5)  # [batch_size, nsrc, 1, npatch]
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    # ncc, _ = torch.topk(ncc, 4, dim=1, largest=False)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    return ncc


def function_ssim(pred, gt, window, channel):
    ntotpx, nviews, nc, h, w = pred.shape
    flat_pred = pred.view(-1, nc, h, w)
    mu1 = F.conv2d(flat_pred, window, padding=0, groups=channel).view(ntotpx, nviews, nc)
    mu2 = F.conv2d(gt, window, padding=0, groups=channel).view(ntotpx, nc)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2).unsqueeze(1)
    mu1_mu2 = mu1 * mu2.unsqueeze(1)

    sigma1_sq = F.conv2d(flat_pred * flat_pred, window, padding=0, groups=channel).view(ntotpx, nviews, nc) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, window, padding=0, groups=channel).view(ntotpx, 1, 3) - mu2_sq
    sigma12 = F.conv2d((pred * gt.unsqueeze(1)).view(-1, nc, h, w), window, padding=0, groups=channel).view(ntotpx, nviews, nc) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    values = 1 - ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return torch.sum(values, dim=2) / 2


def main(args):
    img_0 = cv.imread(args.img0_dir)
    print("img 0 size:", img_0.shape)
    h_0, w_0, _ = img_0.shape
    fix_hw = max(h_0, w_0)
    img_1 = cv.imread(args.img1_dir)
    print("img 1 size:", img_1.shape)
    img_0 = cv.resize(img_0, [fix_hw, fix_hw])
    img_1 = cv.resize(img_1, [fix_hw, fix_hw])

    if args.with_noise:
        img_1 = add_noise_func(img_1, noise_type='gaussian')

    print("The similarity metric is:", args.similarity_metric)
    if args.similarity_metric == "NCC":
        img_0_gray = cv.cvtColor(img_0, cv.COLOR_BGR2GRAY).astype(np.float32)/255.0
        img_1_gray = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY).astype(np.float32)/255.0

        ref_gray = img_0_gray.reshape(1,1,-1,1)
        ref_gray = torch.from_numpy(ref_gray)
        src_gray = img_1_gray.reshape(1,1,-1,1)
        src_gray = torch.from_numpy(src_gray)

        similarity = function_ncc(ref_gray, src_gray)
    elif args.similarity_metric == "SSIM":
        channels = 3
        img_pred = torch.from_numpy(img_0)
        img_gt   = torch.from_numpy(img_1)
        img_pred = img_pred[None,None,:,:,:]
        img_pred = img_pred.permute(0, 1, 4, 2, 3).contiguous() / 255.
        img_gt   = img_gt[None,:,:,:]
        img_gt   = img_gt.permute(0, 3, 1, 2) / 255.
        window_filter = create_window(fix_hw, channels)
        similarity = function_ssim(img_pred, img_gt, window_filter, channels)
    else:
        raise NotImplementedError("Other methods have been implimented!")

    print("The similarity between two patches:", similarity)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img0_dir", type=str, default="000026_sqrt.png", help="image 0 directory")
    parser.add_argument("--img1_dir", type=str, default="000001_sqrt.png", help="image 1 directory")
    parser.add_argument("--similarity_metric", type=str, default="NCC")
    parser.add_argument("--with_noise", action="store_true",
                        default=False, help= "Displaying the inside colors")
    args = parser.parse_args()
    main(args)


