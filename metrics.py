import os
import cv2
import sys
import lpips
import numpy as np
from matplotlib import pyplot as plt
import torch
from utils.loss_utils import ssim
import tempfile
import shutil
from scipy import linalg
from torchvision import models
import torch.nn.functional as F


class LMDMeter:
    def __init__(self, backend='dlib', region='mouth'):
        self.backend = backend
        self.region = region  # mouth or face

        if self.backend == 'dlib':
            import dlib

            # load checkpoint manually
            self.predictor_path = './shape_predictor_68_face_landmarks.dat'
            if not os.path.exists(self.predictor_path):
                raise FileNotFoundError(
                    'Please download dlib checkpoint from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')

            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(self.predictor_path)

        else:

            import face_alignment
            try:
                self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
            except:
                self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

        self.V = 0
        self.N = 0

    def get_landmarks(self, img):

        if self.backend == 'dlib':
            dets = self.detector(img, 1)
            for det in dets:
                shape = self.predictor(img, det)
                # ref: https://github.com/PyImageSearch/imutils/blob/c12f15391fcc945d0d644b85194b8c044a392e0a/imutils/face_utils/helpers.py
                lms = np.zeros((68, 2), dtype=np.int32)
                for i in range(0, 68):
                    lms[i, 0] = shape.part(i).x
                    lms[i, 1] = shape.part(i).y
                break

        else:
            lms = self.predictor.get_landmarks(img)[-1]

        # self.vis_landmarks(img, lms)
        lms = lms.astype(np.float32)

        return lms

    def vis_landmarks(self, img, lms):
        plt.imshow(img)
        plt.plot(lms[48:68, 0], lms[48:68, 1], marker='o', markersize=1, linestyle='-', lw=2)
        plt.show()

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.detach().cpu().numpy()
            inp = (inp * 255).astype(np.uint8)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        # assert B == 1
        preds, truths = self.prepare_inputs(preds[0], truths[0])  # [H, W, 3] numpy array

        # get lms
        lms_pred = self.get_landmarks(preds)
        lms_truth = self.get_landmarks(truths)

        if self.region == 'mouth':
            lms_pred = lms_pred[48:68]
            lms_truth = lms_truth[48:68]

        # avarage
        lms_pred = lms_pred - lms_pred.mean(0)
        lms_truth = lms_truth - lms_truth.mean(0)

        # distance
        dist = np.sqrt(((lms_pred - lms_truth) ** 2).sum(1)).mean(0)

        self.V += dist
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LMD ({self.backend})"), self.measure(), global_step)

    def report(self):
        return f'LMD ({self.backend}) = {self.measure():.6f}'


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, N, 3] or [B, H, W, 3], range in [0, 1]

        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))

        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'


class LPIPSMeter:
    def __init__(self, net='alex', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(truths, preds, normalize=True).item()  # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'


class SSIMMeter:
    def __init__(self, window_size=11):
        self.V = 0
        self.N = 0
        self.window_size = window_size

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                # 确保输入是 [B, C, H, W] 格式
                if inp.dim() == 4 and inp.shape[-1] == 3:  # [B, H, W, 3] -> [B, 3, H, W]
                    inp = inp.permute(0, 3, 1, 2)
                inp = inp.to(torch.float32)  # 确保数据类型正确
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, 3, H, W], range in [0, 1]

        # 计算SSIM
        ssim_value = ssim(truths, preds, window_size=self.window_size, size_average=True)

        self.V += ssim_value.item()
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "SSIM"), self.measure(), global_step)

    def report(self):
        return f'SSIM = {self.measure():.6f}'


class FIDMeter:
    def __init__(self, device=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载Inception v3模型
        self.inception_model = models.inception_v3(pretrained=True)
        self.inception_model.fc = torch.nn.Identity()  # 移除全连接层
        self.inception_model = self.inception_model.eval().to(self.device)

        self.real_activations = []
        self.fake_activations = []

    def clear(self):
        self.real_activations = []
        self.fake_activations = []

    def extract_features(self, x):
        """使用Inception v3模型提取特征"""
        if x.shape[1] != 3:  # 确保通道数在前
            x = x.permute(0, 3, 1, 2)

        # 调整图像大小为299x299 (Inception v3的预期输入)
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        with torch.no_grad():
            features = self.inception_model(x)

        return features.cpu().numpy()

    def update(self, preds, truths):
        """
        更新特征激活值
        preds: 生成图像 [B, H, W, 3]
        truths: 真实图像 [B, H, W, 3]
        """
        # 确保输入在[0, 1]范围内
        preds = torch.clamp(preds, 0, 1)
        truths = torch.clamp(truths, 0, 1)

        # 提取特征
        pred_features = self.extract_features(preds)
        truth_features = self.extract_features(truths)

        # 保存特征
        self.fake_activations.append(pred_features)
        self.real_activations.append(truth_features)

    def calculate_fid(self, act1, act2):
        """计算两个激活值集合之间的FID"""
        act1 = np.vstack(act1)
        act2 = np.vstack(act2)

        # 计算均值和协方差
        mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

        # 计算平方差和
        ssdiff = np.sum((mu1 - mu2) ** 2.0)

        # 计算协方差矩阵的乘积的平方根
        covmean = linalg.sqrtm(sigma1.dot(sigma2))

        # 检查并校正复数结果
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # 计算FID
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def measure(self):
        """计算FID分数"""
        if not self.real_activations or not self.fake_activations:
            return float('inf')

        return self.calculate_fid(self.real_activations, self.fake_activations)

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "FID"), self.measure(), global_step)

    def report(self):
        return f'FID = {self.measure():.6f}'


# 初始化所有的评估器
lmd_meter = LMDMeter(backend='fan')
psnr_meter = PSNRMeter()
lpips_meter = LPIPSMeter()
ssim_meter = SSIMMeter()
fid_meter = FIDMeter()  # 新增FID评估器

# 清空所有评估器
lmd_meter.clear()
psnr_meter.clear()
lpips_meter.clear()
ssim_meter.clear()
fid_meter.clear()

# 读取视频文件
vid_path_1 = sys.argv[1]
vid_path_2 = sys.argv[2]

capture_1 = cv2.VideoCapture(vid_path_1)
capture_2 = cv2.VideoCapture(vid_path_2)

counter = 0
while True:
    ret_1, frame_1 = capture_1.read()
    ret_2, frame_2 = capture_2.read()

    if not ret_1 * ret_2:
        break

    # 转换颜色空间并归一化
    inp_1 = torch.FloatTensor(frame_1[..., ::-1] / 255.0)[None, ...].cuda()
    inp_2 = torch.FloatTensor(frame_2[..., ::-1] / 255.0)[None, ...].cuda()

    # 更新所有评估器
    lmd_meter.update(inp_1, inp_2)
    psnr_meter.update(inp_1, inp_2)
    lpips_meter.update(inp_1, inp_2)
    ssim_meter.update(inp_1, inp_2)
    fid_meter.update(inp_1, inp_2)  # 更新FID评估器

    counter += 1
    if counter % 100 == 0:
        print(f"Processed {counter} frames...")

# 输出所有评估结果
print(lmd_meter.report())
print(psnr_meter.report())
print(lpips_meter.report())
print(ssim_meter.report())
print(fid_meter.report())  # 输出FID结果
