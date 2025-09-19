#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

def patchify(input, patch_size):
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0,2,1).view(-1, 3, patch_size, patch_size)
    return patches

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

#new--------来着DNG


def patchify2(input, patch_size):
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0, 2, 1).view(-1,
                                                                                               1 * patch_size * patch_size)
    return patches


def margin_l2_loss(network_output, gt, margin, return_mask=False):
    mask = (network_output - gt).abs() > margin
    if not return_mask:
        return ((network_output - gt)[mask] ** 2).mean()
    else:
        return ((network_output - gt)[mask] ** 2).mean(), mask


def patch_norm_mse_loss(input, target, patch_size, margin, return_mask=False):
    # Ensure input and target are 4D tensors
    input = input.unsqueeze(0).unsqueeze(0) if input.dim() == 2 else input.unsqueeze(0) if input.dim() == 3 else input
    target = target.unsqueeze(0).unsqueeze(0) if target.dim() == 2 else target.unsqueeze(
        0) if target.dim() == 3 else target

    input_patches = normalize(patchify2(input, patch_size))
    target_patches = normalize(patchify2(target, patch_size))

    # Ensure input_patches and target_patches have the same shape
    min_length = min(input_patches.size(0), target_patches.size(0))
    input_patches = input_patches[:min_length]
    target_patches = target_patches[:min_length]

    return margin_l2_loss(input_patches, target_patches, margin, return_mask)


def patch_norm_mse_loss_global(input, target, patch_size, margin, return_mask=False):
    # Ensure input and target are 4D tensors
    input = input.unsqueeze(0).unsqueeze(0) if input.dim() == 2 else input.unsqueeze(0) if input.dim() == 3 else input
    target = target.unsqueeze(0).unsqueeze(0) if target.dim() == 2 else target.unsqueeze(
        0) if target.dim() == 3 else target

    input_patches = normalize(patchify2(input, patch_size), std=input.std().detach())
    target_patches = normalize(patchify2(target, patch_size), std=target.std().detach())

    # Ensure input_patches and target_patches have the same shape
    min_length = min(input_patches.size(0), target_patches.size(0))
    input_patches = input_patches[:min_length]
    target_patches = target_patches[:min_length]

    return margin_l2_loss(input_patches, target_patches, margin, return_mask)

def patch_norm_mse_loss2(input, target, patch_size, margin, return_mask=False):
    # Ensure input and target are 4D tensors
    input = input.unsqueeze(0).unsqueeze(0) if input.dim() == 2 else input.unsqueeze(0) if input.dim() == 3 else input
    target = target.unsqueeze(0).unsqueeze(0) if target.dim() == 2 else target.unsqueeze(
        0) if target.dim() == 3 else target

    input_patches = normalize(patchify(input, patch_size))
    target_patches = normalize(patchify(target, patch_size))

    # Ensure input_patches and target_patches have the same shape
    min_length = min(input_patches.size(0), target_patches.size(0))
    input_patches = input_patches[:min_length]
    target_patches = target_patches[:min_length]

    return margin_l2_loss(input_patches, target_patches, margin, return_mask)


def patch_norm_mse_loss_global2(input, target, patch_size, margin, return_mask=False):
    # Ensure input and target are 4D tensors
    input = input.unsqueeze(0).unsqueeze(0) if input.dim() == 2 else input.unsqueeze(0) if input.dim() == 3 else input
    target = target.unsqueeze(0).unsqueeze(0) if target.dim() == 2 else target.unsqueeze(
        0) if target.dim() == 3 else target

    input_patches = normalize(patchify(input, patch_size), std=input.std().detach())
    target_patches = normalize(patchify(target, patch_size), std=target.std().detach())

    # Ensure input_patches and target_patches have the same shape
    min_length = min(input_patches.size(0), target_patches.size(0))
    input_patches = input_patches[:min_length]
    target_patches = target_patches[:min_length]

    return margin_l2_loss(input_patches, target_patches, margin, return_mask)

def loss_depth_smoothness(depth, img):
    # 计算图像的梯度
    img_grad_x = img[:, :, :-1] - img[:, :, 1:]
    img_grad_y = img[:, :-1, :] - img[:, 1:, :]

    # 计算权重
    weight_x = torch.exp(-torch.abs(img_grad_x).mean(1, keepdim=True))
    weight_y = torch.exp(-torch.abs(img_grad_y).mean(1, keepdim=True))

    # 调整 weight_x 和 weight_y 的维度为 4D
    weight_x = weight_x.unsqueeze(1)  # 增加 channel 维度
    weight_y = weight_y.unsqueeze(1)  # 增加 channel 维度

    # 使用插值调整 weight_x 和 weight_y 的尺寸，确保与 depth 切片匹配
    weight_x = F.interpolate(weight_x, size=(depth.shape[2] - 1, depth.shape[3]), mode='bilinear', align_corners=False)
    weight_y = F.interpolate(weight_y, size=(depth.shape[2], depth.shape[3]), mode='bilinear', align_corners=False)

    # 插值后去除多余的维度
    weight_x = weight_x.squeeze(1)
    weight_y = weight_y.squeeze(1)

    # 计算深度图平滑损失
    loss = (((depth[:, :, :-1] - depth[:, :, 1:]).abs() * weight_x).sum() +
            ((depth[:, :-1, :] - depth[:, 1:, :]).abs() * weight_y).sum()) / \
           (weight_x.sum() + weight_y.sum())

    return loss

