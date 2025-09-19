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
from torch_ema import ExponentialMovingAverage
from random import randint
from utils.loss_utils import l1_loss, l2_loss, patchify, ssim, patch_norm_mse_loss_global, patch_norm_mse_loss
from gaussian_renderer import render, render_motion
import sys, copy
from scene_pretrain import Scene, GaussianModel, MotionNetwork, PersonalizedMotionNetwork
from utils.general_utils import safe_state
import lpips
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from tensorboardX import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             share_audio_net):
    data_list = [
        # "pretrain/1","pretrain/2"
        "pretrain/Obama1", "pretrain/4", "pretrain/Shaheen", "pretrain/3", "5"
    ]

    testing_iterations = [i * len(data_list) for i in range(0, opt.iterations + 1, 2000)]
    checkpoint_iterations = saving_iterations = [i * len(data_list) for i in range(0, opt.iterations + 1, 10000)] + [
        opt.iterations * len(data_list)]

    # vars
    warm_step = 1000 * len(data_list)  # 预热阶段结束点
    # warm_step = 2000 * len(data_list)  # 预热阶段结束点
    opt.densify_until_iter = (opt.iterations - 1000) * len(data_list)  # 高斯密度化停止点
    lpips_start_iter = 99999999 * len(data_list)  # LPIPS损失启动点（极大值表示不使用）
    motion_stop_iter = opt.iterations * len(data_list)  # 运动网络训练停止点
    mouth_select_iter = (opt.iterations - 10000) * len(data_list)  # 嘴部运动选择结束点
    p_motion_start_iter = 0  # 个性化运动网络启动点
    mouth_step = 1 / max(mouth_select_iter, 1)  # 嘴部运动采样步长
    hair_mask_interval = 7  # 头发掩码更新间隔
    select_interval = 15  # 嘴部运动采样间隔

    opt.iterations *= len(data_list)  # 调整总迭代次数（每个epoch处理所有人物数据）

    first_iter = 0

    tb_writer = prepare_output_and_logger(dataset)  # 准备TensorBoard日志记录器
    if share_audio_net:
        motion_net = MotionNetwork(args=dataset).cuda()
        # 配置优化器（网格学习率5e-3，MLP学习率5e-4）
        motion_optimizer = torch.optim.AdamW(motion_net.get_params(5e-3, 5e-4), betas=(0.9, 0.99), eps=1e-8)
        # 分段学习率调度器
        scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: (
                0.5 ** (iter / mouth_select_iter)) if iter < mouth_select_iter else 0.1 ** (iter / opt.iterations))
        # 指数移动平均（稳定训练）
        ema_motion_net = ExponentialMovingAverage(motion_net.parameters(), decay=0.995)

    scene_list = []  # 初始化场景列表
    for data_name in data_list:
        gaussians = GaussianModel(dataset)  # 创建高斯模型
        _dataset = copy.deepcopy(dataset)  # 克隆数据集配置
        _dataset.source_path = os.path.join(dataset.source_path, data_name)  # 设置当前人物数据路径
        _dataset.model_path = os.path.join(dataset.model_path, data_name)  # 设置当前人物的模型保存路径

        os.makedirs(_dataset.model_path, exist_ok=True)
        with open(os.path.join(_dataset.model_path, "cfg_args"), 'w') as cfg_log_f:  # 将当前配置写入cfg_args文件（便于复现）
            cfg_log_f.write(str(Namespace(**vars(_dataset))))  # Namespace将字典转换为命令行参数格式

        scene = Scene(_dataset, gaussians)  # 创建场景对象（包含相机和高斯模型）
        # 共享音频特征提取网络（身份无关预训练核心）
        if share_audio_net:
            gaussians.neural_motion_grid.audio_net = motion_net.audio_net  # 使用共享的音频特征提取器
            gaussians.neural_motion_grid.audio_att_net = motion_net.audio_att_net  # 使用共享的音频注意力网络
        scene_list.append(scene)  # 将场景添加到场景列表
        gaussians.training_setup(opt)  # 设置高斯模型优化器（学习率、权重衰减等）

    # 非共享音频网络时的独立初始化
    if not share_audio_net:
        motion_net = MotionNetwork(args=dataset).cuda()  # 为每个场景创建独立的运动网络
        # 运动网络优化器（网格参数5e-3，MLP参数5e-4） 提升学习率
        # motion_optimizer = torch.optim.AdamW(motion_net.get_params(5e-3, 5e-4), betas=(0.9, 0.99), eps=1e-8)
        motion_optimizer = torch.optim.AdamW(motion_net.get_params(5e-3, 5e-4), betas=(0.9, 0.99),
                                             eps=1e-8)  # 网格提升2倍+MLP提升2倍
        # 学习率调度器（分段衰减策略）
        scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: (
                0.5 ** (iter / mouth_select_iter)) if iter < mouth_select_iter else 0.1 ** (iter / opt.iterations))
        # 指数移动平均（稳定训练）
        ema_motion_net = ExponentialMovingAverage(motion_net.parameters(), decay=0.995)

    lpips_criterion = lpips.LPIPS(net='alex').eval().cuda()  # 初始化LPIPS感知损失（AlexNet主干）

    bg_color = [0, 1, 0]  # [1, 1, 1] # if dataset.white_background else [0, 0, 0] # 设置绿色背景（[0,1,0]便于抠像）
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # 转换为GPU张量

    # 创建CUDA事件用于精确计时
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None  # 初始化视角栈（用于管理训练相机）
    ema_loss_for_log = 0.0  # 指数平均损失（用于进度条显示）
    progress_bar = tqdm(range(first_iter, opt.iterations), ascii=True, dynamic_ncols=True,
                        desc="Training progress")  # 初始化进度条（动态调整宽度）
    first_iter += 1  # 调整起始迭代（从1开始计数）
    # 主训练循环
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()  # 记录迭代开始时间

        cur_scene_idx = randint(0, len(scene_list) - 1)  # 随机选择场景（人物）
        scene = scene_list[cur_scene_idx]  # 获取场景对象
        gaussians = scene.gaussians  # 获取场景的高斯模型

        gaussians.update_learning_rate(iteration)  # 更新高斯模型学习率（自适应衰减）

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # 每1000次迭代增加球谐阶数（提升表达能力）
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera  获取随机相机视角
        # if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))  # 随机弹出视角

        # find a big mouth
        # 获取嘴部运动边界（从预处理数据）
        mouth_global_lb = viewpoint_cam.talking_dict['mouth_bound'][0]
        mouth_global_ub = viewpoint_cam.talking_dict['mouth_bound'][1]

        mouth_global_lb += (mouth_global_ub - mouth_global_lb) * 0.2  # 动态调整嘴部采样范围
        mouth_window = (mouth_global_ub - mouth_global_lb) * 0.2  # 计算嘴部采样窗口大小

        mouth_lb = mouth_global_lb + mouth_step * iteration * (mouth_global_ub - mouth_global_lb)  # 计算当前迭代的嘴部下界
        mouth_ub = mouth_lb + mouth_window  # 计算当前迭代的嘴上界
        mouth_lb = mouth_lb - mouth_window  # 扩展下界（确保覆盖范围）

        # 设置动作单元（AU）的全局范围
        au_global_lb = 0  # 下界
        au_global_ub = 1  # 上界
        au_window = 0.3  # 采样窗口大小

        au_lb = au_global_lb + mouth_step * iteration * (au_global_ub - au_global_lb)  # 计算当前迭代的AU下界
        au_ub = au_lb + au_window  # 计算当前迭代的AU上界
        au_lb = au_lb - au_window * 0.5  # 扩展AU下界（确保覆盖范围）

        # 预热阶段：选择嘴部开合度在特定范围内的视角
        if iteration < warm_step and iteration < mouth_select_iter:
            if iteration % select_interval == 0:  # 按选择间隔进行采样
                # 当嘴部开合度超出目标范围时
                while viewpoint_cam.talking_dict['mouth_bound'][2] < mouth_lb or \
                        viewpoint_cam.talking_dict['mouth_bound'][2] > mouth_ub:
                    if not viewpoint_stack:  # 如果视角栈空则重新加载
                        viewpoint_stack = scene.getTrainCameras().copy()
                    # 随机选择新视角
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # 主训练阶段：选择眨眼动作在特定范围内的视角
        if warm_step < iteration < mouth_select_iter:
            # 按选择间隔进行采样
            if iteration % select_interval == 0:
                # 当眨眼动作超出目标范围时
                while viewpoint_cam.talking_dict['blink'] < au_lb or viewpoint_cam.talking_dict['blink'] > au_ub:
                    if not viewpoint_stack:  # 如果视角栈空则重新加载
                        viewpoint_stack = scene.getTrainCameras().copy()
                    # 随机选择新视角
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        # 调试模式设置（当迭代达到指定点时启用）
        if (iteration - 1) == debug_from:
            pipe.debug = True

        face_mask = torch.as_tensor(viewpoint_cam.talking_dict["face_mask"]).cuda()  # 面部
        hair_mask = torch.as_tensor(viewpoint_cam.talking_dict["hair_mask"]).cuda()  # 头发
        mouth_mask = torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda()  # 嘴部
        face_mask = face_mask + mouth_mask
        head_mask = face_mask + hair_mask  # 组合头部掩码（面部+头发）

        # 嘴部掩码精细化处理（训练后期启用）
        if iteration > lpips_start_iter:
            max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 创建最大池化层（形态学膨胀操作）

        # 头发掩码更新条件（特定训练阶段且非更新间隔 每七次迭代）
        hair_mask_iter = (warm_step < iteration < lpips_start_iter - 1000) and iteration % hair_mask_interval != 0

        # 预热 静态渲染
        if iteration < warm_step:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        # 单做通用
        elif iteration < p_motion_start_iter:
            render_pkg = render_motion(viewpoint_cam, gaussians, motion_net, pipe, background, return_attn=True)

        # 3. 个性化运动阶段：带个性化运动渲染（运动对齐自适应）
        else:
            render_pkg = render_motion(viewpoint_cam, gaussians, motion_net, pipe, background, return_attn=True,
                                       personalized=True)
        # 解包渲染结果
        image_white, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
            "alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


        # 准备GT图像
        gt_image = viewpoint_cam.original_image.cuda() / 255.0  # 原始图像归一化
        gt_image_white = gt_image * head_mask + background[:, None, None] * ~head_mask  # 应用头部掩码

        # 运动网络冻结（训练后期）
        if iteration > motion_stop_iter:
            for param in motion_net.parameters():
                param.requires_grad = False  # 停止UMF梯度更新

        # Loss
        # 区域特定处理（头发和嘴部）
        if hair_mask_iter:
            image_white[:, hair_mask] = background[:, None]  # 头发区域置为背景
            gt_image_white[:, hair_mask] = background[:, None]  # GT头发区域同样处理


        # 基础重建损失
        Ll1 = l1_loss(image_white, gt_image_white)  # L1像素损失
        DSSIM = opt.lambda_dssim * (1.0 - ssim(image_white, gt_image_white))  # + DSSIM结构相似性损失
        loss = Ll1 + DSSIM
        #############################################################
        # 反向传播前，记录总 Loss 到 TensorBoard
        if tb_writer is not None:
            tb_writer.add_scalar("L1 Loss", Ll1.item(), global_step=iteration)
            tb_writer.add_scalar("DSSIM Loss", DSSIM.item(), global_step=iteration)
        #############################################################

        # 运动正则化（预热阶段后启用）
        if iteration > warm_step:
            # 通用运动场(UMF)正则化
            motion_d_xyz = 1e-5 * (render_pkg['motion']['d_xyz'].abs()).mean()  # 位置变化正则
            loss += motion_d_xyz
            motion_d_rot = 1e-5 * (render_pkg['motion']['d_rot'].abs()).mean()  # 旋转变化正则
            loss += motion_d_rot
            motion_d_opa = 1e-5 * (render_pkg['motion']['d_opa'].abs()).mean()  # 透明度变化正则
            loss += motion_d_opa
            motion_d_scale = 1e-5 * (render_pkg['motion']['d_scale'].abs()).mean()  # 尺度变化正则
            loss += motion_d_scale

            # 头部掩码正则化（前景背景分离）
            head_mask_reg = 1e-3 * (((1 - alpha) * head_mask).mean() + (alpha * ~head_mask).mean())
            loss += head_mask_reg

            #############################################################
            # 反向传播前，记录正则化Loss 到 TensorBoard
            if tb_writer is not None:
                tb_writer.add_scalar("motion_d_xyz", motion_d_xyz.item(), global_step=iteration)
                tb_writer.add_scalar("motion_d_rot", motion_d_rot.item(), global_step=iteration)
                tb_writer.add_scalar("motion_d_opa", motion_d_opa.item(), global_step=iteration)
                tb_writer.add_scalar("motion_d_scale", motion_d_scale.item(), global_step=iteration)
                tb_writer.add_scalar("head_mask_reg", head_mask_reg.item(), global_step=iteration)
            #############################################################

            # 个性化运动阶段（运动对齐自适应）
            if iteration > p_motion_start_iter:
                # 个性化运动正则化
                p_motion_d_xyz = 1e-5 * (render_pkg['p_motion']['d_xyz'].abs()).mean()
                loss += p_motion_d_xyz
                p_motion_d_rot = 1e-5 * (render_pkg['p_motion']['d_rot'].abs()).mean()
                loss += p_motion_d_rot
                p_motion_d_opa = 1e-5 * (render_pkg['p_motion']['d_opa'].abs()).mean()
                loss += p_motion_d_opa
                p_motion_d_scale = 1e-5 * (render_pkg['p_motion']['d_scale'].abs()).mean()
                loss += p_motion_d_scale

                #############################################################
                # 反向传播前，记录正则化Loss 到 TensorBoard
                if tb_writer is not None:
                    tb_writer.add_scalar("p_motion_d_xyz", p_motion_d_xyz.item(), global_step=iteration)
                    tb_writer.add_scalar("p_motion_d_rot", p_motion_d_rot.item(), global_step=iteration)
                    tb_writer.add_scalar("p_motion_d_opa", p_motion_d_opa.item(), global_step=iteration)
                    tb_writer.add_scalar("p_motion_d_scale", p_motion_d_scale.item(), global_step=iteration)
                #############################################################

                # Contrast
                # 对比损失 NC Loss
                audio_feat = viewpoint_cam.talking_dict["auds"].cuda()  # 音频特征
                exp_feat = viewpoint_cam.talking_dict["au_exp"].cuda()  # 表情特征
                # 获取当前场景的个性化运动预测
                p_motion_preds = gaussians.neural_motion_grid(gaussians.get_xyz, audio_feat, exp_feat)
                contrast_loss = 0

                # 跨场景对比学习
                for tmp_scene_idx in range(len(scene_list)):
                    if tmp_scene_idx == cur_scene_idx: continue  # 跳过当前场景
                    with torch.no_grad():
                        tmp_scene = scene_list[tmp_scene_idx]
                        tmp_gaussians = tmp_scene.gaussians
                        # 获取其他场景的预测作为负样本
                        tmp_p_motion_preds = tmp_gaussians.neural_motion_grid(gaussians.get_xyz, audio_feat, exp_feat)
                    # 计算对比损失（点积）
                    contrast_loss_i = (tmp_p_motion_preds['d_xyz'] * p_motion_preds['d_xyz']).sum(-1)
                    contrast_loss_i[contrast_loss_i < 0] = 0  # 截断负值
                    contrast_loss += contrast_loss_i.mean()
                # 添加对比损失
                loss += contrast_loss
                #############################################################
                # 反向传播前，记录NC Loss 到 TensorBoard
                if tb_writer is not None:
                    tb_writer.add_scalar("contrast_loss", contrast_loss.item(), global_step=iteration)
                #############################################################


            # 嘴部注意力正则化
            [xmin, xmax, ymin, ymax] = viewpoint_cam.talking_dict['lips_rect']  # 嘴唇矩形区域
            # 通用运动场嘴部注意力正则
            mouth_attn_reg = 5e-3 * (render_pkg["attn"][1, xmin:xmax, ymin:ymax]).mean()
            loss += mouth_attn_reg
            #############################################################
            # 反向传播前，记录Loss 到 TensorBoard
            if tb_writer is not None:
                tb_writer.add_scalar("mouth_attn_reg", mouth_attn_reg.item(), global_step=iteration)
            #############################################################

            # 个性化运动场嘴部注意力正则
            if iteration > p_motion_start_iter:
                p_mouth_attn_reg = 5e-3 * (render_pkg["p_attn"][1, xmin:xmax, ymin:ymax]).mean()
                loss += p_mouth_attn_reg
                #############################################################
                # 反向传播前，记录Loss 到 TensorBoard
                if tb_writer is not None:
                    tb_writer.add_scalar("p_mouth_attn_reg", p_mouth_attn_reg.item(), global_step=iteration)
                #############################################################

            # 头发区域注意力正则化（非头发掩码更新时
            if not hair_mask_iter:
                au_attn = 1e-4 * (render_pkg["attn"][1][hair_mask]).mean()  # 音频注意力
                loss += au_attn
                ex_attn = 1e-4 * (render_pkg["attn"][0][hair_mask]).mean()  # 表情注意力
                loss += ex_attn
                #############################################################
                # 反向传播前，记录Loss 到 TensorBoard
                if tb_writer is not None:
                    tb_writer.add_scalar("au_attn", au_attn.item(), global_step=iteration)
                    tb_writer.add_scalar("ex_attn", ex_attn.item(), global_step=iteration)
                #############################################################



        # 损失反向传播
        loss.backward()
        # 记录迭代结束时间点（用于计算迭代耗时）
        iter_end.record()

        # 进入无梯度计算环境（节省显存）
        with torch.no_grad():
            # Progress bar
            # 计算指数移动平均(EMA)损失（用于进度条显示）
            # 0.4:当前损失权重, 0.6:历史损失权重（平滑显示）
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            # 每10次迭代更新一次进度条，以及关闭进度条
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}",
                                          "Mouth": f"{mouth_lb:.{1}f}-{mouth_ub:.{1}f}"})  # , "AU25": f"{au_lb:.{1}f}-{au_ub:.{1}f}"
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # 训练报告生成（日志记录和验证）
            training_report(
                tb_writer,  # TensorBoard写入器
                iteration,  # 当前迭代次数
                Ll1,  # L1损失值
                loss,  # 总损失值
                l1_loss,  # L1损失函数
                iter_start.elapsed_time(iter_end),  # 迭代耗时(ms)
                testing_iterations,  # 测试迭代点列表
                scene,  # 当前场景
                motion_net,  # 运动网络
                # 动态选择渲染函数：预热阶段用render，其他用render_motion
                render if iteration < warm_step else render_motion,
                (pipe, background)  # 渲染管线参数
            )
            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(str(iteration)+'_face')

            # 检查点保存（在预定迭代时）
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                ckpt = (motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
                torch.save(ckpt, dataset.model_path + "/chkpnt_face_latest" + ".pth")
                with ema_motion_net.average_parameters():
                    ckpt_ema = (motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
                    torch.save(ckpt, dataset.model_path + "/chkpnt_ema_face_latest" + ".pth")
                for _scene in scene_list:
                    _gaussians = _scene.gaussians
                    ckpt = (_gaussians.capture(), motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
                    torch.save(ckpt, _scene.model_path + "/chkpnt_face_" + str(iteration) + ".pth")
                    torch.save(ckpt, _scene.model_path + "/chkpnt_face_latest" + ".pth")

            # Densification
            # 高斯密度控制（在指定迭代范围内）
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                # 更新图像空间中高斯点的最大半径（用于后续剪枝）
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                # 添加密度统计信息（基于可见性和视图空间位置）
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold,
                                                0.05 + 0.25 * iteration / opt.densify_until_iter, scene.cameras_extent,
                                                size_threshold)

            # bg prune
            # 背景剪枝（与密度化同步进行）
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                from utils.sh_utils import eval_sh  # 导入球谐函数评估工具

                # 准备球谐系数（变换维度：[N, C, sh_dim] -> [N, 3, sh_dim]）
                shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
                # 计算视线方向（世界坐标到相机中心）
                dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
                # 归一化视线方向
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                # 评估球谐函数获取RGB颜色
                sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                # 计算最终颜色并截断到[0.5,1.0] -> [0,0.5]映射到0
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                # 创建背景颜色掩码（检测绿色背景：R<30, G>225, B<30）
                bg_color_mask = (colors_precomp[..., 0] < 30 / 255) * (colors_precomp[..., 1] > 225 / 255) * (
                        colors_precomp[..., 2] < 30 / 255)
                gaussians.prune_points(bg_color_mask.squeeze())  # 执行背景点剪枝（移除绿色背景点）

            # Optimizer step
            # 优化器更新（未达到最终迭代时）
            if iteration < opt.iterations:
                # 运动网络优化器步进（更新UMF参数）
                motion_optimizer.step()
                # 高斯模型优化器步进（更新高斯点参数）
                gaussians.optimizer.step()

                # 清零运动网络梯度
                motion_optimizer.zero_grad()
                # 清零高斯模型梯度（set_to_none=True节省内存）
                gaussians.optimizer.zero_grad(set_to_none=True)

                # 更新学习率调度器
                scheduler.step()
                # 更新运动网络的指数移动平均（EMA）
                ema_motion_net.update()


# 检查模型路径是否已设置
def prepare_output_and_logger(args):
    # 检查模型路径是否已设置
    if not args.model_path:
        # 尝试从环境变量获取OAR任务ID
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            # 生成唯一UUID作为路径标识
            unique_str = str(uuid.uuid4())
        # 创建输出路径（使用唯一字符串前10字符）
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # 设置输出文件夹
    print("Output folder: {}".format(args.model_path))  # 打印输出路径
    os.makedirs(args.model_path, exist_ok=True)  # 创建目录（已存在时不报错）

    # 将配置参数写入cfg_args文件
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        # Namespace对象转字符串保存（包含所有参数）
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 创建TensorBoard写入器
    tb_writer = None
    # 检查TensorBoard是否可用
    if TENSORBOARD_FOUND:
        # 初始化SummaryWriter（日志写入指定路径）
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")  # 警告信息

    # 返回TensorBoard写入器
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, motion_net,
                    renderFunc, renderArgs):
    # TensorBoard日志记录
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)  # L1损失
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)  # 总损失
        tb_writer.add_scalar('iter_time', elapsed, iteration)  # 迭代耗时(ms)

    # Report test and samples of training set
    # 验证配置：在指定测试迭代点执行
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # 创建验证配置（测试集+训练集）测试集配置：每5帧取1帧，共20帧  训练集配置：每5帧取1帧，共6帧
        validation_configs = ({'name': 'test',
                               'cameras': [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in
                                           range(5, 100, 5)]},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})
        # 创建存储图像的根目录
        report_root = os.path.join('training_reports', f'iter_{iteration}')
        os.makedirs(report_root, exist_ok=True)
        # 遍历验证配置
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                # 初始化L1和PSNR
                l1_test = 0.0
                psnr_test = 0.0
                # 遍历配置中的每个相机视角
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg_p = None  # 初始化个性化渲染包
                    # 根据渲染函数类型选择渲染方式
                    if renderFunc is render:  # 静态渲染
                        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    else:
                        # 运动渲染（UMF）
                        render_pkg = renderFunc(viewpoint, scene.gaussians, motion_net, return_attn=True, frame_idx=0,
                                                *renderArgs)
                        # 运动渲染（个性化）
                        render_pkg_p = renderFunc(viewpoint, scene.gaussians, motion_net, return_attn=True, frame_idx=0,
                                                  personalized=True, *renderArgs)

                    # 图像后处理
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)  # 裁剪到[0,1]
                    alpha = render_pkg["alpha"]  # 透明度通道
                    normal = render_pkg["normal"] * 0.5 + 0.5  # 法线图归一化

                    # image = image - renderArgs[1][:, None, None] * (1.0 - alpha) + background[:, None, None].cuda() / 255.0 * (1.0 - alpha)
                    image = image
                    # gt_image = torch.clamp(viewpoint.original_image.to("cuda") / 255.0, 0.0, 1.0)

                    # 准备GT图像（考虑透明度）
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda") / 255.0, 0.0, 1.0) * alpha + renderArgs[
                        1][:,
                    None,
                    None] * (
                                       1.0 - alpha)

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

                    # 嘴部掩码精细化
                    mouth_mask = torch.as_tensor(viewpoint.talking_dict["mouth_mask"]).cuda()
                    max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                    mouth_mask_post = (-max_pool(-max_pool(mouth_mask[None].float())))[0].bool()

                    # TensorBoard可视化（前10个视角）
                    if tb_writer and (idx < 10):
                        # 基础渲染结果
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        # 个性化渲染结果
                        if render_pkg_p is not None:
                            tb_writer.add_images(config['name'] + "_view_{}/render_p".format(viewpoint.image_name),
                                                 render_pkg_p['render'][None], global_step=iteration)
                        # 真实图像
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                             gt_image[None], global_step=iteration)
                        # tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), (render_pkg["depth"] / render_pkg["depth"].max())[None], global_step=iteration)
                        # 精细化嘴部掩码
                        tb_writer.add_images(config['name'] + "_view_{}/mouth_mask_post".format(viewpoint.image_name),
                                             (~mouth_mask_post * gt_image)[None], global_step=iteration)
                        # 原始嘴部掩码
                        tb_writer.add_images(config['name'] + "_view_{}/mouth_mask".format(viewpoint.image_name),
                                             (~mouth_mask[None] * gt_image)[None], global_step=iteration)
                        # 法线图
                        tb_writer.add_images(config['name'] + "_view_{}/normal".format(viewpoint.image_name),
                                             normal[None], global_step=iteration)
                        # tb_writer.add_images(config['name'] + "_view_{}/normal_mono".format(viewpoint.image_name), (viewpoint.talking_dict["normal"]*0.5+0.5)[None], global_step=iteration)

                        # 运动渲染特有可视化
                        if renderFunc is not render:
                            # 音频注意力图
                            tb_writer.add_images(config['name'] + "_view_{}/attn_a".format(viewpoint.image_name),
                                                 (render_pkg["attn"][0] / render_pkg["attn"][0].max())[None, None],
                                                 global_step=iteration)
                            # 表情注意力图
                            tb_writer.add_images(config['name'] + "_view_{}/attn_e".format(viewpoint.image_name),
                                                 (render_pkg["attn"][1] / render_pkg["attn"][1].max())[None, None],
                                                 global_step=iteration)
                            # 个性化注意力图
                            if render_pkg_p is not None:
                                tb_writer.add_images(config['name'] + "_view_{}/p_attn_a".format(viewpoint.image_name),
                                                     (render_pkg_p["p_attn"][0] / render_pkg_p["p_attn"][0].max())[
                                                         None, None], global_step=iteration)
                                tb_writer.add_images(config['name'] + "_view_{}/p_attn_e".format(viewpoint.image_name),
                                                     (render_pkg_p["p_attn"][1] / render_pkg_p["p_attn"][1].max())[
                                                         None, None], global_step=iteration)

                    # 计算评估指标L1和PSNR
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                # 计算平均指标
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                # print评估结果
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # 场景统计分析
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)  # 不透明度直方图
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)  # 高斯点总数
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
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument('--share_audio_net', action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.share_audio_net)

    # All done
    print("\nTraining complete.")