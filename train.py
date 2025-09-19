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
import numpy as np
from PIL import Image
import os
import random
import torch
from random import randint

from matplotlib import pyplot as plt

from utils.loss_utils import l1_loss, l2_loss, patchify, ssim, normalize, patch_norm_mse_loss, \
    patch_norm_mse_loss_global, patch_norm_mse_loss_global2, patch_norm_mse_loss2
from gaussian_renderer import render, render_motion, render_motion_opa, render_motion_depth
import sys
from scene import Scene, GaussianModel, MotionNetwork
from utils.general_utils import safe_state
import lpips
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.normal_utils import depth_to_normal
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torch
torch.autograd.set_detect_anomaly(True)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, mode_long, pretrain_ckpt_path):
    testing_iterations = [1] + [i for i in range(0, opt.iterations + 1, 2000)] # 设置测试迭代点：从第1次开始，每10000次迭代测试一次
    # 设置检查点和保存点：每10000次迭代保存，加上最终迭代
    checkpoint_iterations =  saving_iterations = [i for i in range(0, opt.iterations + 1, 2000)] + [opt.iterations]

    # vars
    # 训练阶段控制变量
    warm_step = 3000  # 预热阶段结束点
    opt.densify_until_iter = opt.iterations - 3000  # 高斯密度化停止点
    bg_iter = opt.iterations  # 背景切换点
    lpips_start_iter = opt.densify_until_iter - 1500  # LPIPS损失启动点
    motion_stop_iter = bg_iter  # 运动网络训练停止点
    mouth_select_iter = opt.iterations  # 嘴部运动选择结束点
    mouth_step = 1 / max(mouth_select_iter, 1)  # 嘴部运动采样步长
    hair_mask_interval = 7  # 头发掩码更新间隔
    select_interval = 10  # 嘴部运动采样间隔

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    if tb_writer:
        print("True")
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians)

    # 创建运动网络
    motion_net = MotionNetwork(args=dataset).cuda()
    # 配置优化器（分层学习率）
    motion_optimizer = torch.optim.AdamW(motion_net.get_params(5e-3, 5e-4), betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: 0.1 if iter < warm_step else 0.5 ** (iter / opt.iterations))
    if mode_long:   # 长训练模式调整
        scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: 0.1 if iter < warm_step else 0.1 ** (iter / opt.iterations))

    # Load pre-trained
    # 加载预训练运动网络
    (motion_params, _, _) = torch.load(pretrain_ckpt_path)
    # gaussians.restore(model_params, opt)
    motion_net.load_state_dict(motion_params)

    # (model_params, _, _, _) = torch.load(os.path.join("output/pretrain4/macron/chkpnt_face_latest.pth"))
    # gaussians.neural_motion_grid.load_state_dict(model_params[-1])

    lpips_criterion = lpips.LPIPS(net='alex').eval().cuda()    # 初始化LPIPS感知损失

    gaussians.training_setup(opt)   # 设置高斯模型优化器
    if checkpoint:  # 恢复训练检查点
        (model_params, motion_params, motion_optimizer_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        motion_net.load_state_dict(motion_params)
        motion_optimizer.load_state_dict(motion_optimizer_params)

    if not mode_long:    # 短训练模式配置
        gaussians.max_sh_degree = 1 # 限制球谐函数阶数

    # 绿色背景配置（用于抠像）
    bg_color = [0, 1, 0]   # [1, 1, 1] # if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 训练计时器
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # 训练进度监控
    viewpoint_stack = None  # 视角数据栈
    ema_loss_for_log = 0.0  # 指数移动平均损失
    progress_bar = tqdm(range(first_iter, opt.iterations), ascii=True, dynamic_ncols=True, desc="Training progress")
    first_iter += 1  # 起始迭代号调整

    # 在训练循环之前初始化损失记录
    loss_records = {
        'total': [],
        'l1': [],
        'dssim': [],
        'normal_consistency': [],
        'depth_consistency': [],
        'motion_d_xyz': [],
        'motion_d_rot': [],
        'motion_d_opa': [],
        'motion_d_scale': [],
        'motion_p_xyz': [],
        'opacity_reg': [],
        'attn_lips': [],
        'attn_audio': [],
        'attn_expression': [],
        'lpips': [],
        'iterations': []
    }

    for iteration in range(first_iter, opt.iterations + 1):         # 主训练循环开始
        iter_start.record() # 记录迭代开始时间点（CUDA事件）
        # 更新高斯模型的学习率（自适应衰减策略）
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # 每1000次迭代增加球谐函数阶数（提升细节表现能力）
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()   # 提升球谐函数阶数

        # Pick a random Camera
        # 获取训练视角
        if not viewpoint_stack: # 检查视角栈是否为空
            viewpoint_stack = scene.getTrainCameras().copy()    # 从场景复制训练视角
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) # 随机选择一个视角

        # find a big mouth
        # 嘴部动作范围计算
        mouth_global_lb = viewpoint_cam.talking_dict['mouth_bound'][0]  # 下界
        mouth_global_ub = viewpoint_cam.talking_dict['mouth_bound'][1]  # 上界
        mouth_global_lb += (mouth_global_ub - mouth_global_lb) * 0.2  # 调整下界
        mouth_window = (mouth_global_ub - mouth_global_lb) * 0.5  # 采样窗口大小

        # 动态嘴部采样范围
        # mouth_step * iteration实现随训练推进扩大的采样范围
        mouth_lb = mouth_global_lb + mouth_step * iteration * (mouth_global_ub - mouth_global_lb)
        mouth_ub = mouth_lb + mouth_window
        mouth_lb = mouth_lb - mouth_window  # 扩展下界

        # 眨眼动作(AU)范围设置
        au_global_lb = 0  # 下界
        au_global_ub = 1  # 上界
        au_window = 0.4  # 采样窗口

        # 动态眨眼采样范围
        au_lb = au_global_lb + mouth_step * iteration * (au_global_ub - au_global_lb)
        au_ub = au_lb + au_window
        au_lb = au_lb - au_window * 1.5  # 扩展下界

        if iteration < warm_step and iteration < mouth_select_iter:
            if iteration % select_interval == 0:
                # 筛选嘴部开口度在目标范围内的视角
                while viewpoint_cam.talking_dict['mouth_bound'][2] < mouth_lb or viewpoint_cam.talking_dict['mouth_bound'][2] > mouth_ub:
                    if not viewpoint_stack:
                        viewpoint_stack = scene.getTrainCameras().copy()
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


        if warm_step < iteration < mouth_select_iter:
            # 筛选眨眼程度在目标范围内的视角
            if iteration % select_interval == 0:
                while viewpoint_cam.talking_dict['blink'] < au_lb or viewpoint_cam.talking_dict['blink'] > au_ub:
                    if not viewpoint_stack:
                        viewpoint_stack = scene.getTrainCameras().copy()
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))



        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True   # 启用渲染管线调试
        # 加载面部/头发/嘴部掩码
        face_mask = torch.as_tensor(viewpoint_cam.talking_dict["face_mask"]).cuda()
        hair_mask = torch.as_tensor(viewpoint_cam.talking_dict["hair_mask"]).cuda()
        mouth_mask = torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda()
        face_mask = face_mask + mouth_mask #
        head_mask = face_mask + hair_mask
        FACE_STAGE_ITER = opt.iterations - 2000

        if iteration <= FACE_STAGE_ITER:

        # 嘴部掩码精细化处理
            if iteration <= lpips_start_iter:
                max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

            # 头发掩码更新控制
            hair_mask_iter = (warm_step < iteration < lpips_start_iter - 1000) and iteration % hair_mask_interval != 0

            # 预热阶段渲染
            if iteration < warm_step:
                # render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                enable_align = iteration > 1000 # 1000次迭代后启用对齐
                render_pkg = render_motion(viewpoint_cam, gaussians, motion_net, pipe, background, return_attn=True, personalized=False, align=enable_align)

            # 主训练阶段渲染
            else:
                # 始终启用对齐
                render_pkg = render_motion(viewpoint_cam, gaussians, motion_net, pipe, background, return_attn=True, personalized=False, align=True)

            image_white, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # 准备GT图像
            gt_image  = viewpoint_cam.original_image.cuda() / 255.0  # 归一化
            gt_image_white = gt_image * head_mask + background[:, None, None] * ~head_mask   # 背景替换

            # 运动网络冻结（训练后期）
            if iteration > motion_stop_iter:
                for param in motion_net.parameters():
                    param.requires_grad = False # 停止梯度更新

            # 高斯参数冻结（最终阶段）
            if iteration > bg_iter:
                gaussians._xyz.requires_grad = False
                gaussians._opacity.requires_grad = False
                gaussians._scaling.requires_grad = False
                gaussians._rotation.requires_grad = False



            # Loss
            if iteration < bg_iter:
                if hair_mask_iter:  # 判断头发掩码更新周期
                    image_white[:, hair_mask] = background[:, None] # 将渲染图像的头发区域像素值替换为背景色
                    gt_image_white[:, hair_mask] = background[:, None]  # 将真实图像的头发区域像素值替换为背景色


                patch_range = (10,30)  # 10 - 30
                loss_l2_dpt = patch_norm_mse_loss2(image_white[None, ...], gt_image_white[None, ...],
                                                  randint(patch_range[0], patch_range[1]), 0.02)
                patch_img = 0.0004 * loss_l2_dpt
                loss_global = patch_norm_mse_loss_global2(image_white[None, ...], gt_image_white[None, ...],
                                                         randint(patch_range[0], patch_range[1]), 0.02)
                global_img = 0.004 * loss_global

                Ll1 = l1_loss(image_white, gt_image_white)  # 计算L1 Loss
                loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image_white, gt_image_white))   # 组合L1和DSSIM损失
                loss += global_img + patch_img


                if not mode_long and iteration > warm_step + 2000:  # 短模式+预热后2000次迭代

                    render_normal = render_pkg["normal"]
                    gt_normal = viewpoint_cam.talking_dict["normal"].cuda()

                    normal_loss = 0.01 * (1 -gt_normal * render_normal ).sum(0)[head_mask].mean()
                    loss += normal_loss

                    if tb_writer is not None:
                        tb_writer.add_scalar(" normal_loss", normal_loss.item(), global_step=iteration)
                    if iteration % opt.opacity_reset_interval > 100:    # 深度一致性损失
                        depth = render_pkg["depth"][0]  # 渲染深度
                        depth_mono = viewpoint_cam.talking_dict['depth'].cuda() # 单目估计深度

                        depth_loss= 1e-2 * (normalize(depth)[face_mask] - normalize(depth_mono)[face_mask]).abs().mean()
                        loss += depth_loss
                        if tb_writer is not None:
                            tb_writer.add_scalar("depth_loss", depth_loss.item(), global_step=iteration)


                #运动变化正则化
                if iteration > warm_step:
                    loss += 1e-5 * (render_pkg['motion']['d_xyz'].abs()).mean()  # 位移变化正则
                    loss += 1e-5 * (render_pkg['motion']['d_rot'].abs()).mean()  # 旋转变化正则
                    loss += 1e-5 * (render_pkg['motion']['d_opa'].abs()).mean()  # 透明度变化正则
                    loss += 1e-5 * (render_pkg['motion']['d_scale'].abs()).mean()  # 尺度变化正则
                    loss += 1e-5 * (render_pkg['p_motion']['p_xyz'].abs()).mean()  # 个性化位移正则

                    # 透明度正则化。 确保头部区域内不透明，区域外透明
                    opacity_loss = 1e-3 * (((1-alpha) * head_mask).mean() + (alpha * ~head_mask).mean())
                    loss += opacity_loss

                    if tb_writer is not None:
                        tb_writer.add_scalar("opacity_loss", opacity_loss.item(), global_step=iteration)

                    [xmin, xmax, ymin, ymax] = viewpoint_cam.talking_dict['lips_rect']  # 嘴唇矩形区域
                    attn_lips= 1e-4 * (render_pkg["attn"][1, xmin:xmax, ymin:ymax]).mean() # 增强嘴部注意力
                    loss += attn_lips
                    if tb_writer is not None:
                        tb_writer.add_scalar("attn_lips", attn_lips.item(), global_step=iteration)

                    if not hair_mask_iter:  # 非头发掩码更新时
                        attn_aud= 1e-4 * (render_pkg["attn"][1][hair_mask]).mean()    # 音频注意力
                        loss += attn_aud
                        attn_exp= 1e-4 * (render_pkg["attn"][0][hair_mask]).mean()    # 表情注意力
                        loss += attn_exp
                        if tb_writer is not None:
                            tb_writer.add_scalar("attn_aud", attn_aud.item(), global_step=iteration)
                            tb_writer.add_scalar("attn_exp", attn_exp.item(), global_step=iteration)
                    # loss += l2_loss(image_white[:, xmin:xmax, ymin:ymax], image_white[:, xmin:xmax, ymin:ymax])

                # patch-image Depth Loss
                if iteration > 1000:
                    depth, alpha = render_pkg["depth"], render_pkg["alpha"]
                    depth_mono = viewpoint_cam.talking_dict['depth'].cuda() # 单目估计深度
                    depth_mono[~(face_mask+hair_mask)] = 0
                    patch_range = (10, 30)  #10 - 30
                    loss_l2_dpt = patch_norm_mse_loss(depth[None, ...], depth_mono[None, ...],
                                                      randint(patch_range[0], patch_range[1]), 0.02)
                    patch_depth= 0.0004 * loss_l2_dpt

                    loss_global = patch_norm_mse_loss_global(depth[None, ...], depth_mono[None, ...],
                                                             randint(patch_range[0], patch_range[1]), 0.02)
                    global_depth= 0.004 * loss_global

                    loss += patch_depth
                    loss += global_depth
                    if tb_writer is not None:
                        tb_writer.add_scalar("patch_depth", patch_depth.item(), global_step=iteration)
                        tb_writer.add_scalar("global_depth", global_depth.item(), global_step=iteration)

                image_t = image_white.clone()
                gt_image_t = gt_image_white.clone()



            else:   # 超过bg_iter后切换到真实背景
                # with real bg
                # 合成真实背景图像
                image = image_white - background[:, None, None] * (1.0 - alpha) + viewpoint_cam.background.cuda() / 255.0 * (1.0 - alpha)

                # 计算真实背景下的损失
                Ll1 = l1_loss(image, gt_image)
                loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

                image_t = image.clone()
                gt_image_t = gt_image.clone()

            if iteration > lpips_start_iter:   # LPIPS损失启动点
                # mask mouth
                # 嘴部区域掩码
                [xmin, xmax, ymin, ymax] = viewpoint_cam.talking_dict['lips_rect']
                if mode_long:   # 长训练模式嘴部强化
                    loss += 0.01 * lpips_criterion(image_t.clone()[:, xmin:xmax, ymin:ymax] * 2 - 1, gt_image_t.clone()[:, xmin:xmax, ymin:ymax] * 2 - 1).mean()

                # 随机补丁LPIPS损失
                patch_size = random.randint(32, 48) * 2 # 64-96像素补丁
                if mode_long:   # 长训练模式增强
                    loss += 0.2 * lpips_criterion(patchify(image_t[None, ...] * 2 - 1, patch_size), patchify(gt_image_t[None, ...] * 2 - 1, patch_size)).mean()
                # 基础LPIPS损失
                lpips_loss= 0.01 * lpips_criterion(patchify(image_t[None, ...] * 2 - 1, patch_size), patchify(gt_image_t[None, ...] * 2 - 1, patch_size)).mean()
                loss += lpips_loss
                if tb_writer is not None:
                    tb_writer.add_scalar("lpips_loss", lpips_loss.item(), global_step=iteration)


            loss_records['iterations'].append(iteration)

        else:
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            # 渲染面部模型：render_motion
            render_pkg = render_motion(viewpoint_cam, gaussians, motion_net, pipe, background, align=True)
            # 提取渲染结果的关键信息
            viewspace_point_tensor, visibility_filter = render_pkg["viewspace_points"], render_pkg["visibility_filter"]

            # 透明度提取
            alpha = render_pkg["alpha"]
            image = render_pkg["render"] - background[:, None, None] * (
                        1.0 - alpha) + viewpoint_cam.background.cuda() / 255.0 * (1.0 - alpha)

            # 获取归一化真实图像
            gt_image = viewpoint_cam.original_image.cuda() / 255.0
            # 创建带背景的真实图像（头部区域保留，其他区域置为背景色）
            gt_image_white = gt_image * head_mask + background[:, None, None] * ~head_mask

            # 参数冻结控制
            if iteration > bg_iter:
                for param in motion_net.parameters():
                    param.requires_grad = False

                # 冻结几何参数梯度
                gaussians._xyz.requires_grad = False
                gaussians._scaling.requires_grad = False
                gaussians._rotation.requires_grad = False


            # Loss
            # 损失计算（绿幕阶段）
            if iteration < bg_iter:
                image[:, ~head_mask] = background[:, None]  # 非头部区域置为背景色

                Ll1 = l1_loss(image, gt_image_white)  # 计算L1损失
                loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image_white))  # 添加DSSIM结构损失
                loss += 1e-3 * (((1 - alpha) * head_mask).mean() + (alpha * ~head_mask).mean())  # 添加透明度正则项

                # 克隆图像
                image_t = image.clone()
                gt_image_t = gt_image_white.clone()

            # 损失计算（真实背景阶段）
            else:
                Ll1 = l1_loss(image, gt_image)  # 直接计算L1损失
                loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))  # 添加DSSIM结构损失

                # 克隆图像
                image_t = image.clone()
                gt_image_t = gt_image.clone()

            if iteration > lpips_start_iter:  # 迭代超过总迭代数一半
                patch_size = random.randint(16, 21) * 2  # 随机生成补丁大小（32-42像素）
                # 计算补丁级LPIPS感知损失    加权(0.05)加入总损失
                loss += 0.05 * lpips_criterion(patchify(image_t[None, ...] * 2 - 1, patch_size),
                                               patchify(gt_image_t[None, ...] * 2 - 1, patch_size)).mean()



        loss.backward()  # 反向传播计算梯度
        iter_end.record()  # 记录迭代结束时间

        with torch.no_grad():
            # Progress bar 进度条
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log   # 更新指数移动平均损失
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}", "Mouth": f"{mouth_lb:.{1}f}-{mouth_ub:.{1}f}"}) # , "AU25": f"{au_lb:.{1}f}-{au_ub:.{1}f}"
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # 训练报告生成
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, motion_net, render if iteration < warm_step else render_motion, (pipe, background))
            if (iteration in saving_iterations):  # 在保存点迭代
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(str(iteration) + '_face')  # 保存高斯模型

            if (iteration in checkpoint_iterations):  # 在检查点迭代
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                # 准备检查点数据
                ckpt = (
                    gaussians.capture(),  # 捕获高斯模型状态
                    motion_net.state_dict(),  # 运动网络参数
                    motion_optimizer.state_dict(),  # 优化器状态
                    iteration  # 当前迭代次数
                )
                # 保存带迭代编号的检查点
                torch.save(ckpt, scene.model_path + "/chkpnt_" + str(iteration) + ".pth")
                # 保存最新检查点
                torch.save(ckpt, scene.model_path + "/chkpnt_latest" + ".pth")
                torch.save(loss_records,  scene.model_path + "/loss_records" + ".pth")

            # Densification 密度控制
            if iteration < opt.densify_until_iter:  # 在密度控制阶段
                # Keep track of max radii in image-space for pruning
                # 更新最大半径（用于剪枝）
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)    # 添加密度统计信息

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:   # 密度化与剪枝
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05 + 0.25 * iteration / opt.densify_until_iter, scene.cameras_extent, size_threshold)

                if not mode_long:   # 短训练模式
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            # bg prune 背景剪枝
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                from utils.sh_utils import eval_sh

                shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
                dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

                bg_color_mask = (colors_precomp[..., 0] < 30/255) * (colors_precomp[..., 1] > 225/255) * (colors_precomp[..., 2] < 30/255)
                gaussians.prune_points(bg_color_mask.squeeze())

                if not mode_long:   # 短训练模式额外剪枝
                    gaussians.prune_points((gaussians.get_xyz[:, -1] < -0.07).squeeze())    # 剪除深度过小的点


            # Optimizer step    优化器步骤
            if iteration <= opt.iterations:
                if iteration <= FACE_STAGE_ITER:
                    motion_optimizer.step()
                    gaussians.optimizer.step()
                    motion_optimizer.zero_grad()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    scheduler.step()
                else:
                    # 执行优化步骤
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)



def prepare_output_and_logger(args):    # 输出目录与日志
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, motion_net, renderFunc, renderArgs):
    if tb_writer:  # TensorBoard可用时
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)  # L1损失
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)  # 总损失
        tb_writer.add_scalar('iter_time', elapsed, iteration)  # 迭代耗时

    # Report test and samples of training set
    # 报告测试情况以及训练集样本
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # 创建验证集配置   测试集5-100帧每5帧取1帧     训练集5-30帧每5帧取1帧
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(5, 100, 5)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        # 创建存储图像的根目录
        report_root = os.path.join('training_reports/train', f'iter_{iteration}')
        os.makedirs(report_root, exist_ok=True)
        for config in validation_configs:   # 遍历测试/训练配置
            if config['cameras'] and len(config['cameras']) > 0:    # 检查相机数据存在
                l1_test = 0.0
                psnr_test = 0.0

                # 遍历配置中的每个相机视角
                for idx, viewpoint in enumerate(config['cameras']):
                    if renderFunc is render:     # 基础渲染
                        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    else:   # 运动渲染
                        render_pkg = renderFunc(viewpoint, scene.gaussians, motion_net, return_attn=True, frame_idx=0, align=True, *renderArgs)

                    # 图像后处理
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)  # 裁剪到[0,1]
                    alpha = render_pkg["alpha"]  # 透明度通道
                    normal = render_pkg["normal"] * 0.5 + 0.5   #法线图处理 归一化到[0,1]

                    # 深度图处理 归一化
                    depth = render_pkg["depth"] * alpha + (render_pkg["depth"] * alpha).mean() * (1 - alpha)
                    depth = (depth - depth.min()) / (depth.max() - depth.min())

                    # 深度法线图生成
                    depth_normal = depth_to_normal(viewpoint, render_pkg["depth"]).permute(2, 0, 1)  # 深度转法线
                    depth_normal = depth_normal * alpha.detach()  # 应用透明度
                    depth_normal = depth_normal * 0.5 + 0.5  # 归一化到[0,1]

                    # 背景合成
                    image = image - renderArgs[1][:, None, None] * (1.0 - alpha) + viewpoint.background.cuda() / 255.0 * (1.0 - alpha)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda") / 255.0, 0.0, 1.0) # GT图像

                    # === 添加图像保存代码 ===
                    # 创建配置特定目录
                    config_name = config['name']
                    config_dir = os.path.join(report_root, config_name)
                    os.makedirs(config_dir, exist_ok=True)

                    # 安全文件名
                    safe_name = viewpoint.image_name.replace('/', '_')

                    # 保存渲染图像
                    render_img = image.permute(1, 2, 0).cpu().numpy()
                    render_img = (render_img * 255).astype(np.uint8)
                    render_path = os.path.join(config_dir, f"{safe_name}_render.png")
                    Image.fromarray(render_img).save(render_path)

                    # 保存GT图像
                    gt_img = gt_image.permute(1, 2, 0).cpu().numpy()
                    gt_img = (gt_img * 255).astype(np.uint8)
                    gt_path = os.path.join(config_dir, f"{safe_name}_gt.png")
                    Image.fromarray(gt_img).save(gt_path)
                    # === 结束添加代码 ===


                    # 嘴部掩码处理
                    mouth_mask = torch.as_tensor(viewpoint.talking_dict["mouth_mask"]).cuda()
                    max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                    mouth_mask_post = (-max_pool(-max_pool(mouth_mask[None].float())))[0].bool()  # 形态学闭运算

                    if tb_writer and (idx < 5): # 只记录前5个视角
                        # 基础渲染结果
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        # 真实图像
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        # 深度图
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        # 嘴部掩码可视化
                        tb_writer.add_images(config['name'] + "_view_{}/mouth_mask_post".format(viewpoint.image_name), (~mouth_mask_post * gt_image)[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/mouth_mask".format(viewpoint.image_name), (~mouth_mask[None] * gt_image)[None], global_step=iteration)
                        # 法线图
                        tb_writer.add_images(config['name'] + "_view_{}/normal".format(viewpoint.image_name), normal[None], global_step=iteration)
                        # 深度法线图
                        tb_writer.add_images(config['name'] + "_view_{}/normal_from_depth".format(viewpoint.image_name), depth_normal[None], global_step=iteration)

                        # 运动渲染特有可视化
                        if renderFunc is not render:
                            # 音频注意力图
                            tb_writer.add_images(config['name'] + "_view_{}/attn_a".format(viewpoint.image_name), (render_pkg["attn"][0] / render_pkg["attn"][0].max())[None, None], global_step=iteration)
                            # 表情注意力图
                            tb_writer.add_images(config['name'] + "_view_{}/attn_e".format(viewpoint.image_name), (render_pkg["attn"][1] / render_pkg["attn"][1].max())[None, None], global_step=iteration)

                    # 计算当前视角指标
                    l1_test += l1_loss(image, gt_image).mean().double()  # L1损失
                    psnr_test += psnr(image, gt_image).mean().double()  # PSNR

                # 计算平均指标
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)  # 不透明度直方图
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)   # 高斯点数量统计
        torch.cuda.empty_cache()




if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--long", action='store_true', default=False)
    parser.add_argument("--pretrain_path", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.long, args.pretrain_path)

    # All done
    print("\nTraining complete.")
