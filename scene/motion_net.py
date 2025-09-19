import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder


# 新增频率编码器（参考频率版）
def get_embedder(multires, i=3):
    if multires == 0:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x: embedder_obj.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, leakyReLU=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        if leakyReLU:
            self.act = nn.LeakyReLU(0.02)
        else:
            self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


# Audio feature extractor
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [1, seq_len, dim_aud]
        y = x.permute(0, 2, 1)  # [1, dim_aud, seq_len]
        y = self.attentionConvNet(y)
        y = self.attentionNet(y.view(1, self.seq_len)).view(1, self.seq_len, 1)
        return torch.sum(y * x, dim=1)  # [1, dim_aud]


# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(dim_in, 32 if dim_in < 128 else 128, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32 if dim_in < 128 else 128, 32 if dim_in < 128 else 128, kernel_size=3, stride=2, padding=1,
                      bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32 if dim_in < 128 else 128, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size / 2)
        x = x[:, :, 8 - half_w:8 + half_w]
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x)
        return x


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        out = self.audio_encoder(x)
        out = out.squeeze(2).squeeze(2)

        return out


# Audio feature extractor
class AudioNet_ave(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet_ave, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.02, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.02, True),
            nn.Linear(128, dim_aud),
        )

    def forward(self, x):
        # half_w = int(self.win_size/2)
        # x = x[:, :, 8-half_w:8+half_w]
        # x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x).permute(1, 0, 2).squeeze(0)
        return x


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden,
                                 self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
                # x = F.dropout(x, p=0.1, training=self.training)

        return x


class MotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim=32,
                 ind_dim=0,
                 args=None,

                 multires=10,  # 频率编码的频率数
                 ):
        super(MotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        elif 'ave' in args.audio_extractor:
            self.audio_in_dim = 32
        else:
            raise NotImplementedError

        self.bound = 0.15
        self.exp_eye = True

        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1)

            # audio network
        self.audio_dim = audio_dim
        if args.audio_extractor == 'ave':
            self.audio_net = AudioNet_ave(self.audio_in_dim, self.audio_dim)
        else:
            self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)
        self.audio_att_net = AudioAttNet(self.audio_dim)

        '''
        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz


        self.num_layers = 3       
        self.hidden_dim = 64
        '''
        # 位置编码（频率编码）
        self.embed_fn, self.xyz_input_ch = get_embedder(multires, i=3)

        self.exp_in_dim = 6 - 1
        self.eye_dim = 6 if self.exp_eye else 0

        self.exp_encode_net = MLP(self.exp_in_dim, self.eye_dim - 1, 16, 2)
        self.eye_att_net = MLP(self.xyz_input_ch, self.eye_dim, 16, 2)

        '''
        self.exp_encode_net = nn.ModuleList([
            nn.Linear(self.exp_in_dim, 16, bias=False),
            nn.Linear(16, self.eye_dim - 1, bias=False)
        ])
        self.eye_att_net = nn.ModuleList([
            nn.Linear(self.xyz_input_ch, 16, bias=False),
            nn.Linear(16, self.eye_dim, bias=False)
        ])
        '''
        '''
        # 更深的网络结构
        self.eye_att_net = nn.ModuleList([
            nn.Linear(self.xyz_input_ch, 64, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.eye_dim, bias=False),
            nn.Sigmoid()  # 添加Sigmoid确保输出在0-1范围内
        ])

        # 更强大的网络
        self.exp_encode_net = nn.Sequential(
            nn.Linear(self.exp_in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 5),  # 确保输出维度是5，因为后面要与e[-1:]拼接得到6维
            nn.Tanh()
        )
        
        # 添加眼部区域的空间偏置
        self.eye_region_bias = nn.Parameter(torch.zeros(1, self.xyz_input_ch))

        # 添加眼部区域检测网络
        self.eye_region_detector = nn.Linear(self.xyz_input_ch, 1)
        self.eye_region_detector = nn.Sequential(
            nn.Linear(self.xyz_input_ch, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        
        # rot: 4   xyz: 3   opac: 1  scale: 3
        self.out_dim = 11
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.eye_dim + self.individual_dim, self.out_dim, self.hidden_dim, self.num_layers)
        
        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)
        
        self.cache = None
        '''
        # 新论文的 MLP 配置
        self.D = 8  # 层数
        self.W = 256  # 隐藏层维度
        self.skips = [self.D // 2]  # skip connection 在第 4 层
        self.input_ch = self.xyz_input_ch + self.audio_dim + self.eye_dim + self.individual_dim  # 输入维度
        self.out_dim = 11  # 输出维度

        # 定义 MLP 层
        self.deform_net = nn.ModuleList(
            [nn.Linear(self.input_ch, self.W)] +  # 第一层
            [nn.Linear(self.W, self.W) if i not in self.skips else nn.Linear(self.W + self.input_ch, self.W)
             for i in range(self.D - 1)]  # 中间层
        )
        # 输出层
        self.gaussian_warp = nn.Linear(self.W, 3)  # 位移
        self.gaussian_rotation = nn.Linear(self.W, 4)  # 旋转
        self.gaussian_opacity = nn.Linear(self.W, 1)  # 不透明度
        self.gaussian_scaling = nn.Linear(self.W, 3)  # 缩放

        # 音频通道注意力
        # self.aud_ch_att_net = MLP(self.xyz_input_ch, self.audio_dim, 32, 2)
        self.aud_ch_att_net = nn.ModuleList([
            nn.Linear(self.xyz_input_ch, 32, bias=False),
            nn.Linear(32, self.audio_dim, bias=False)
        ])
        '''
        # 融合权重
        self.eye_feature_fusion = nn.Sequential(
            nn.Linear(self.xyz_input_ch * 2, self.xyz_input_ch),
            nn.ReLU(inplace=True),
            nn.Linear(self.xyz_input_ch, self.xyz_input_ch),
            nn.Sigmoid()  # 输出门控权重
        )
        '''
    '''
    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:,:1], x[:,-1:]], dim=-1)
        return xy, yz, xz
    '''

    def encode_x(self, xyz):
        '''
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)

        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
        '''
        return self.embed_fn(xyz)  # 使用频率编码

    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        enc_a = self.audio_net(a)  # [1/8, 64]
        enc_a = self.audio_att_net(enc_a.unsqueeze(0))  # [1, 64]

        return enc_a

    def forward(self, x, a, e=None, c=None):
        # x: [N, 3], in [-bound, bound]
        #enc_x = self.encode_x(x, bound=self.bound)
        enc_x = self.encode_x(x)  #位置编码（频率编码）

        # 音频特征
        enc_a = self.encode_audio(a)
        enc_a = enc_a.repeat(enc_x.shape[0], 1)

        # eye_region_mask = self.eye_region_detector(enc_x)
        # enc_x_eye = enc_x + self.eye_region_bias * eye_region_mask
        # eye_region_mask = torch.sigmoid(self.eye_region_detector(enc_x)) * 0.1  # 缩放因子限制影响
        # enc_x_eye = enc_x + self.eye_region_bias * eye_region_mask
        '''
        # 融合原始特征和眼部增强特征
        combined_features = torch.cat([enc_x, enc_x_eye], dim=-1)
        fusion_weights = self.eye_feature_fusion(combined_features)

        # 加权融合 - 在眼部区域更侧重增强特征
        fused_enc_x = enc_x * (1 - fusion_weights) + enc_x_eye * fusion_weights
        '''
        # aud_ch_att = self.aud_ch_att_net(enc_x)
        aud_ch_att = self.aud_ch_att_net[0](enc_x)
        aud_ch_att = F.relu(aud_ch_att, inplace=True)  #11
        aud_ch_att = self.aud_ch_att_net[1](aud_ch_att)  #1
        enc_w = enc_a * aud_ch_att

        # 眼睛特征
        #eye_att = torch.relu(self.eye_att_net(enc_x))
        #enc_e = self.exp_encode_net(e[:-1])

        # eye_att = self.eye_att_net[0](enc_x)
        # eye_att = F.relu(eye_att, inplace=True)
        # eye_att = torch.relu(self.eye_att_net[1](eye_att))
        eye_att = torch.relu(self.eye_att_net(enc_x))
        #eye_att = self.eye_att_net[0](enc_x_eye)
        #for i in range(1, len(self.eye_att_net)):
        #    eye_att = self.eye_att_net[i](eye_att)

        # enc_e = self.exp_encode_net[0](e[:-1])
        # enc_e = F.relu(enc_e, inplace=True)
        # enc_e = self.exp_encode_net[1](enc_e)
        enc_e = self.exp_encode_net(e[:-1])
        enc_e = torch.cat([enc_e, e[-1:]], dim=-1)
        enc_e = enc_e * eye_att

        # 合并输入
        if c is not None:
            c = c.repeat(enc_x.shape[0], 1)
            h = torch.cat([enc_x, enc_w, enc_e, c], dim=-1)
        else:
            h = torch.cat([enc_x, enc_w, enc_e], dim=-1)

        # MLP 前向传播（新论文的 8 层 MLP）
        for i, l in enumerate(self.deform_net):
            h = self.deform_net[i](h)
            h = F.relu(h, inplace=True)
            if i in self.skips:
                h = torch.cat([h, enc_x, enc_w, enc_e], dim=-1) if c is None else torch.cat(
                    [h, enc_x, enc_w, enc_e, c], dim=-1)
        '''
        for i, l in enumerate(self.deform_net):
            if i in self.skips:
                h = torch.cat([h, enc_x, enc_w, enc_e], dim=-1) if c is None else \
                    torch.cat([h, enc_x, enc_w, enc_e, c], dim=-1)

            h = self.deform_net[i](h)
            if i != len(self.deform_net) - 1:  # 最后一层通常不激活
                h = F.relu(h, inplace=True)
        '''
        # 输出
        d_xyz = self.gaussian_warp(h) * 1e-2
        d_rot = self.gaussian_rotation(h)
        d_opa = self.gaussian_opacity(h)
        d_scale = self.gaussian_scaling(h)
        results = {
            'd_xyz': d_xyz,
            'd_rot': d_rot,
            'd_opa': d_opa,
            'd_scale': d_scale,
            'ambient_aud': aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye': eye_att.norm(dim=-1, keepdim=True),
        }
        '''
        h = self.sigma_net(h)
        d_xyz = h[..., :3] * 1e-2
        d_rot = h[..., 3:7]
        d_opa = h[..., 7:8]
        d_scale = h[..., 8:11]
        results = {
            'd_xyz': d_xyz,
            'd_rot': d_rot,
            'd_opa': d_opa,
            'd_scale': d_scale,
            'ambient_aud': aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye': eye_att.norm(dim=-1, keepdim=True),
        }
        '''
        self.cache = results
        return results

    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):
        '''
        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        '''
        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.deform_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.gaussian_warp.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.gaussian_rotation.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.gaussian_opacity.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.gaussian_scaling.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})

        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.eye_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.exp_encode_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        # params.append({'params': self.eye_region_bias, 'lr': lr_net * 0.05, 'weight_decay': 0})
        # params.append({'params': self.eye_region_detector.parameters(), 'lr': lr_net * 0.1, 'weight_decay': wd})
        # params.append({'params': self.eye_feature_fusion.parameters(), 'lr': lr_net * 0.05, 'weight_decay': wd})
        return params




class PersonalizedMotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim=32,
                 ind_dim=0,
                 args=None,
                 multires=10,  # 频率编码的频率数
                 ):
        super(PersonalizedMotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        elif 'ave' in args.audio_extractor:
            self.audio_in_dim = 32
        else:
            raise NotImplementedError

        self.args = args
        self.bound = 0.15
        self.exp_eye = args.type == 'face'

        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1)

            # audio network
        self.audio_dim = audio_dim
        if args.audio_extractor == 'ave':
            self.audio_net = AudioNet_ave(self.audio_in_dim, self.audio_dim)
        else:
            self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)
        self.audio_att_net = AudioAttNet(self.audio_dim)

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz


        self.num_layers = 3

        # 位置编码（频率编码）
        self.embed_fn, self.xyz_input_ch = get_embedder(multires, i=3)

        self.exp_in_dim = 6 - 1
        self.eye_dim = 6 if self.exp_eye else 0
        if self.exp_eye:
            self.exp_encode_net = MLP(self.exp_in_dim, self.eye_dim - 1, 16, 2)
            self.eye_att_net = MLP(self.in_dim, self.eye_dim, 16, 2)


        # 新论文的 MLP 配置
        self.D = 8  # 层数
        self.W = 256  # 隐藏层维度
        self.skips = [self.D // 2]  # skip connection 在第 4 层
        #self.input_ch = self.xyz_input_ch + self.audio_dim + self.eye_dim + self.individual_dim  # 输入维度
        self.input_ch = self.in_dim + self.audio_dim + self.eye_dim + self.individual_dim  # 输入维度
        self.out_dim = 11 if args.type == 'face' else 7  # 输出维度

        # 定义 MLP 层
        self.deform_net = nn.ModuleList(
            [nn.Linear(self.input_ch, self.W)] +  # 第一层
            [nn.Linear(self.W, self.W) if i not in self.skips else nn.Linear(self.W + self.input_ch, self.W)
             for i in range(self.D - 1)]  # 中间层
        )
        # 输出层
        self.gaussian_warp = nn.Linear(self.W, 3)  # 位移
        self.gaussian_rotation = nn.Linear(self.W, 4)  # 旋转
        self.gaussian_opacity = nn.Linear(self.W, 1)  # 不透明度
        self.gaussian_scaling = nn.Linear(self.W, 3)  # 缩放

        # 音频通道注意力
        # self.aud_ch_att_net = MLP(self.xyz_input_ch, self.audio_dim, 32, 2)
        self.aud_ch_att_net = nn.ModuleList([
            #nn.Linear(self.xyz_input_ch, 32, bias=False),
            nn.Linear(self.in_dim, 32, bias=False),
            nn.Linear(32, self.audio_dim, bias=False)
        ])

        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.eye_dim + self.individual_dim, self.out_dim,
                             self.hidden_dim, self.num_layers)
        self.align_net = MLP(self.in_dim, 6, self.hidden_dim, 2)


    def encode_x(self, xyz):

        return self.embed_fn(xyz)  # 使用频率编码


    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        enc_a = self.audio_net(a)  # [1/8, 64]
        enc_a = self.audio_att_net(enc_a.unsqueeze(0))  # [1, 64]

        return enc_a

    def forward(self, x, a, e=None, c=None):
        enc_x = self.encode_x(x)  # 位置编码（频率编码）

        enc_a = self.encode_audio(a)
        enc_a = enc_a.repeat(enc_x.shape[0], 1)


        aud_ch_att = self.aud_ch_att_net[0](enc_x)
        aud_ch_att = F.relu(aud_ch_att, inplace=True)  # 1
        aud_ch_att = self.aud_ch_att_net[1](aud_ch_att)  # 1
        enc_w = enc_a * aud_ch_att
        h = torch.cat([enc_x, enc_w], dim=-1)

        # 初始化输入特征字典
        input_features = {
            'h': h,
            'enc_x': enc_x,
            'enc_w': enc_w
        }
        # 眼睛特征
        if self.exp_eye:
            eye_att = torch.relu(self.eye_att_net(enc_x))

            enc_e = self.exp_encode_net(e[:-1])

            enc_e = torch.cat([enc_e, e[-1:]], dim=-1)
            input_features['enc_e'] = enc_e * eye_att
            enc_e = enc_e * eye_att
            h = torch.cat([enc_x, enc_w, enc_e], dim=-1)

        if c is not None:
            # 确保c有正确的形状
            if c.dim() == 1:
                c = c.unsqueeze(0)
            input_features['c'] = c.repeat(enc_x.shape[0], 1)
            h = torch.cat([h, c], dim=-1)


        # MLP 前向传播（新论文的 8 层 MLP）
        for i, l in enumerate(self.deform_net):
            h = self.deform_net[i](h)
            h = F.relu(h, inplace=True)
            if i in self.skips:
                # 安全的拼接：只使用存在的特征
                to_concat = [h]
                for key in ['enc_x', 'enc_w', 'enc_e', 'c']:
                    if key in input_features:
                        to_concat.append(input_features[key])
                h = torch.cat(to_concat, dim=-1)


        # 输出
        d_xyz = self.gaussian_warp(h) * 1e-2
        d_rot = self.gaussian_rotation(h)

        if self.args.type == "face":
            d_opa = self.gaussian_opacity(h)
            d_scale = self.gaussian_scaling(h)
        else:
            d_opa = d_scale = None


        p = self.align_net(enc_x)
        p_xyz = p[..., :3] * 1e-2
        p_scale = torch.tanh(p[..., 3:] / 5) * 0.25 + 1

        return {
            'd_xyz': d_xyz,
            'd_rot': d_rot,
            'd_opa': d_opa,
            'd_scale': d_scale,
            'ambient_aud': aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye': eye_att.norm(dim=-1, keepdim=True) if self.exp_eye else None,
            'p_xyz': p_xyz,
            'p_scale': p_scale,
        }

    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        params = [
            {'params': self.audio_net.parameters(), 'name': 'neural_audio_net', 'lr': lr_net, 'weight_decay': wd},
            {'params': self.encoder_xy.parameters(), 'name': 'neural_encoder_xy', 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'name': 'neural_encoder_xy', 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'name': 'neural_encoder_xy', 'lr': lr},
            {'params': self.deform_net.parameters(), 'name': 'neural_deform_net','lr': lr_net, 'weight_decay': wd},
            {'params': self.align_net.parameters(), 'name': 'neural_align_net', 'lr': lr_net / 2, 'weight_decay': wd},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'name': 'neural_audio_att_net', 'lr': lr_net * 5,
                       'weight_decay': 0.0001})
        if self.individual_dim > 0:
            params.append(
                {'params': self.individual_codes, 'name': 'neural_individual_codes', 'lr': lr_net, 'weight_decay': wd})

        params.append({'params': self.aud_ch_att_net.parameters(), 'name': 'neural_aud_ch_att_net', 'lr': lr_net,
                       'weight_decay': wd})

        if self.exp_eye:
            params.append({'params': self.eye_att_net.parameters(), 'name': 'neural_eye_att_net', 'lr': lr_net,
                           'weight_decay': wd})
            params.append({'params': self.exp_encode_net.parameters(), 'name': 'neural_exp_encode_net', 'lr': lr_net,
                           'weight_decay': wd})
        return params
