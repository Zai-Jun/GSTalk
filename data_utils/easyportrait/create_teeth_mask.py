# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from mmseg.apis import inference_segmentor, init_segmentor
import os
import glob
from tqdm import tqdm
import numpy as np
import cv2


def main():
    parser = ArgumentParser()
    parser.add_argument('dataset', help='Dataset directory')
    parser.add_argument('--config',
                        default="./data_utils/easyportrait/local_configs/easyportrait_experiments_v2/fpn-fp/fpn-fp.py",
                        help='Config file')
    parser.add_argument('--checkpoint', default="./data_utils/easyportrait/fpn-fp-512.pth", help='Checkpoint file')

    args = parser.parse_args()

    # 初始化模型
    model = init_segmentor(args.config, args.checkpoint, device='cuda:0')

    # 设置路径
    dataset_path = os.path.join(args.dataset, 'ori_imgs')

    # 创建输出目录
    teeth_out_path = os.path.join(args.dataset, 'teeth_mask')
    eye_out_path = os.path.join(args.dataset, 'eye_mask')  # 合并的眼睛掩码目录

    os.makedirs(teeth_out_path, exist_ok=True)
    os.makedirs(eye_out_path, exist_ok=True)

    # 处理所有图像
    for img_path in tqdm(glob.glob(os.path.join(dataset_path, '*.jpg'))):
        # 获取原始图像用于可视化
        original_img = cv2.imread(img_path)

        # 推理获取分割结果
        result = inference_segmentor(model, img_path)
        seg_map = result[0]  # 获取H×W的分割图

        # 生成牙齿掩码 (类别7)
        teeth_mask = np.zeros_like(seg_map)
        teeth_mask[seg_map == 7] = 1

        # 生成眼睛掩码 (合并左眼5和右眼6)
        eye_mask = np.zeros_like(seg_map)
        eye_mask[(seg_map == 4) | (seg_map == 5)] = 1

        # 分离左右眼用于可视化
        left_eye_mask = (seg_map == 4)
        right_eye_mask = (seg_map == 5)

        # 构造输出路径
        base_name = os.path.basename(img_path).replace('.jpg', '')

        # 保存牙齿掩码 (只保存npy)
        np.save(os.path.join(teeth_out_path, f"{base_name}.npy"), teeth_mask.astype(np.bool_))

        # 保存眼睛掩码 (npy和jpg)
        np.save(os.path.join(eye_out_path, f"{base_name}.npy"), eye_mask.astype(np.bool_))

        # 创建眼睛可视化图像
        eye_vis = np.zeros_like(original_img)

        # 左眼用绿色标记
        eye_vis[left_eye_mask] = [0, 255, 0]  # BGR格式的绿色

        # 右眼用红色标记
        eye_vis[right_eye_mask] = [255, 0, 0]  # BGR格式的红色

        # 叠加到原始图像上
        eye_overlay = cv2.addWeighted(original_img, 0.7, eye_vis, 0.3, 0)

        # 保存可视化图像
        cv2.imwrite(os.path.join(eye_out_path, f"{base_name}.jpg"), eye_overlay)


if __name__ == '__main__':
    main()