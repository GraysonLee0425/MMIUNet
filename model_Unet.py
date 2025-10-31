# -*- coding: utf-8 -*-
"""
MultiModalUNet — 修改版
- 明确 4 次下采样（Down1..Down4），对应空间尺度：256 -> 128 -> 64 -> 32 -> 16
- 每个尺度使用 DenseCrossModal（HyperDense-like）与 CrossModalGating
- 保留 TabularProjector (MTF + SHAP) 在最底层 (H/16, W/16) 的广播与融合
- 对称的解码器：4 次上采样 (Up4..Up1)
接口保持不变：
forward(img_input, ms_input=None, ir_input=None, excel=None, shap_pos=None, shap_neg=None, shap_rank=None)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------
# 基本块
# -------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# -------------------
# Up 模块 (清晰通道签名)
# -------------------
class Up(nn.Module):
    def __init__(self, up_in_ch, skip_ch, out_ch):
        super().__init__()
        # 把上采样先降到 out_ch
        self.up = nn.ConvTranspose2d(up_in_ch, out_ch, kernel_size=2, stride=2)
        # conv block 接受 concat 后的通道 (out_ch + skip_ch) -> out_ch
        self.cb = ConvBlock(out_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.cb(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Down module that returns (pooled_conv, skip_pre_pool)
    - skip: 保留未下采样的特征 (用于该尺度的 Inception/CMDM)
    - pool: 下采样并 conv 后的特征 (供下一层 Down 输入)
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skip = x                     # 保留未下采样特征 (用于 DenseCrossModal 的 skip)
        x = self.pool(x)             # 下采样 (H/2, W/2)
        x = self.conv(x)             # 映射到 out_channels
        return x, skip

# -------------------
# InceptionModule（二版）
# -------------------
class InceptionModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 每分支输出近似均分
        b1 = out_ch // 4
        b2 = out_ch // 4
        b3 = out_ch // 4
        b4 = out_ch - (b1 + b2 + b3)
        # branch1: 1x1
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, b1, kernel_size=1, bias=False),
            nn.BatchNorm2d(b1),
            nn.ReLU(inplace=True)
        )
        # branch2: 1x1 -> 3x3
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, b2, kernel_size=1, bias=False),
            nn.BatchNorm2d(b2),
            nn.ReLU(inplace=True),
            nn.Conv2d(b2, b2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(b2),
            nn.ReLU(inplace=True)
        )
        # branch3: 1x1 -> 3x3 (dilated)
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, b3, kernel_size=1, bias=False),
            nn.BatchNorm2d(b3),
            nn.ReLU(inplace=True),
            nn.Conv2d(b3, b3, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(b3),
            nn.ReLU(inplace=True)
        )
        # branch4: avgpool -> 1x1
        self.b4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, b4, kernel_size=1, bias=False),
            nn.BatchNorm2d(b4),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        o1 = self.b1(x)
        o2 = self.b2(x)
        o3 = self.b3(x)
        o4 = self.b4(x)
        out = torch.cat([o1, o2, o3, o4], dim=1)
        return out

# -------------------
# DenseCrossModal (HyperDense-like 简化实现)
# -------------------
class DenseCrossModal(nn.Module):
    def __init__(self, in_chs):
        """
        in_chs: list of channel dims for each modality at this scale e.g. [c,c,c]
        This module concatenates all modality feature maps and maps back to each modality's channels.
        """
        super().__init__()
        self.in_chs = in_chs
        total = sum(in_chs)
        # 为每条模态建立 conv mapping total -> ch_i
        self.out_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(total, ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            ) for ch in in_chs
        ])
    def forward(self, feats):
        # feats: list of tensors same spatial size
        x = torch.cat(feats, dim=1)
        outs = []
        for i, conv in enumerate(self.out_convs):
            out_i = conv(x) + feats[i]
            outs.append(out_i)
        return outs

# -------------------
# CrossModalGating（只用于图像模态融合）
# -------------------
class CrossModalGating(nn.Module):
    def __init__(self, in_chs, out_ch):
        """
        in_chs: list of channel dims for each input at this scale (e.g. [c,c,c])
        out_ch: output channels after fusion (fused feature channels)
        """
        super().__init__()
        total = sum(in_chs)
        # gate network outputs per-channel gate map (same spatial size)
        self.gate_net = nn.Sequential(
            nn.Conv2d(total, total, kernel_size=1, bias=True),
            nn.BatchNorm2d(total),
            nn.ReLU(inplace=True),
            nn.Conv2d(total, total, kernel_size=1),
            nn.Sigmoid()
        )
        # use Inception to fuse gated concat into out_ch
        self.fuse = InceptionModule(total, out_ch)
    def forward(self, feats):
        x = torch.cat(feats, dim=1)
        gates = self.gate_net(x)     # [B, total, H, W]
        gated = x * gates
        out = self.fuse(gated)       # [B, out_ch, H, W]
        return out, gates

# -------------------
# TabularProjector（残差门控版：SHAP门控调节Excel特征）
# - 输入: mtf [B,10], shap_pos/neg/rank [B,10] or [10] or None
# - 输出: map_b [B, emb_dim, H_b, W_b], gate_vec [B, emb_dim]
# -------------------
class TabularProjector(nn.Module):
    def __init__(self, mtf_dim=10, shap_dim=None, emb_dim=512):
        super().__init__()
        self.mtf_dim = mtf_dim
        if shap_dim is None:
            shap_dim = mtf_dim * 3
        self.shap_dim = shap_dim
        self.emb_dim = emb_dim

        # 1. 首先将原始Excel特征投影到更高的维度（基础信息通路）
        self.fc_excel = nn.Linear(mtf_dim, emb_dim)

        # 2. SHAP门控网络: 输入 shap_dim, 输出 emb_dim，用于调节
        self.shap_gate_net = nn.Sequential(
            nn.Linear(shap_dim, emb_dim // 2),  # -> emb_dim//2
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim // 2, emb_dim),
            nn.Sigmoid()  # 输出范围[0,1], 作为调节系数
        )

    def forward(self, mtf, shap_pos=None, shap_neg=None, shap_rank=None, target_hw=(1,1)):
        """
        mtf: [B, mtf_dim] (Excel特征)
        shap_pos/shap_neg/shap_rank: [10] or [B,10] or None
        target_hw: (H_b, W_b)
        returns:
          map_b : [B, emb_dim, H_b, W_b] (空间特征图)
          gate_vec: [B, emb_dim] (门控向量, 用于监控调节强度)
        """
        B = mtf.size(0)

        # 1. 处理 shap_pos/shap_neg/shap_rank（支持广播或 None）
        if shap_pos is None:
            shp_pos = torch.zeros(B, self.mtf_dim, device=mtf.device)
        else:
            shp_pos = shap_pos if shap_pos.dim() == 2 else shap_pos.unsqueeze(0).repeat(B, 1)

        if shap_neg is None:
            shp_neg = torch.zeros(B, self.mtf_dim, device=mtf.device)
        else:
            shp_neg = shap_neg if shap_neg.dim() == 2 else shap_neg.unsqueeze(0).repeat(B, 1)

        if shap_rank is None:
            shp_rank = torch.zeros(B, self.mtf_dim, device=mtf.device)
        else:
            shp_rank = shap_rank if shap_rank.dim() == 2 else shap_rank.unsqueeze(0).repeat(B, 1)

        # concat pos, neg, rank -> [B, 3*mtf_dim]
        shap_comb = torch.cat([shp_pos, shp_neg, shp_rank], dim=1)  # [B, shap_comb_dim]

        # pad or trim to match shap_dim if needed
        if shap_comb.size(1) != self.shap_dim:
            if shap_comb.size(1) < self.shap_dim:
                pad = torch.zeros(B, self.shap_dim - shap_comb.size(1), device=mtf.device)
                shap_comb = torch.cat([shap_comb, pad], dim=1)
            else:
                shap_comb = shap_comb[:, :self.shap_dim]

        # 2. 生成门控向量: SHAP -> Gate [B, emb_dim]
        gate_vec = self.shap_gate_net(shap_comb)  # [B, emb_dim]

        # 3. 处理基础Excel特征通路
        excel_base = self.fc_excel(mtf)  # [B, mtf_dim] -> [B, emb_dim]
        excel_base = F.relu(excel_base)  # 添加非线性

        # 4. 应用残差门控: X_guided = X_base * (1 + Gate)
        excel_guided = excel_base * (1.0 + gate_vec)  # [B, emb_dim]

        # 5. 广播到空间尺寸
        H_b, W_b = target_hw
        map_b = excel_guided.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H_b, W_b).contiguous()  # [B, emb_dim, H_b, W_b]

        return map_b, gate_vec

# -------------------
# 主模型：MultiModalUNet（Inception 二版在图像分支、dense 融合、bottom）
# 修改点：
# - 明确 4 次下采样（Down1..Down4） => scales: H, H/2, H/4, H/8, H/16
# - 在每个尺度都使用 DenseCrossModal + CrossModalGating
# - Bottom 保持 Tabular(SHAP) 融合
# - 对称解码器 4 次上采样
# -------------------
class MultiModalUNet(nn.Module):
    def __init__(self, base_ch=32, mtf_dim=10, use_shap_gate=True, use_separate_img=True):
        """
        base_ch: 基础通道 (c1)
        mtf_dim: 文本特征维度 (10)
        use_separate_img: True => 传入 rgb, ms, ir 三张张量；False => 传入合并9ch张量
        """
        super().__init__()
        self.base = base_ch
        self.mtf_dim = mtf_dim
        self.use_shap_gate = use_shap_gate
        self.use_separate_img = use_separate_img

        # 通道 schedule
        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4
        c4 = base_ch * 8
        c5 = base_ch * 16
        self.scale_chs = [c1, c2, c3, c4, c5]

        # --- image encoders (RGB / MS / IR)，每个层后面接 InceptionModule 增强多感受野（二版）
        # scale1 (H x W)
        self.rgb_in       = ConvBlock(3, c1)
        self.rgb_incep1   = InceptionModule(c1, c1)

        self.ms_in        = ConvBlock(5, c1)
        self.ms_incep1    = InceptionModule(c1, c1)

        self.ir_in        = ConvBlock(1, c1)
        self.ir_incep1    = InceptionModule(c1, c1)

        # scale2 (H/2 x W/2)   -- Down1
        self.rgb_down1    = Down(c1, c2)
        self.rgb_incep2   = InceptionModule(c2, c2)

        self.ms_down1     = Down(c1, c2)
        self.ms_incep2    = InceptionModule(c2, c2)

        self.ir_down1     = Down(c1, c2)
        self.ir_incep2    = InceptionModule(c2, c2)

        # scale3 (H/4 x W/4)   -- Down2
        self.rgb_down2    = Down(c2, c3)
        self.rgb_incep3   = InceptionModule(c3, c3)

        self.ms_down2     = Down(c2, c3)
        self.ms_incep3    = InceptionModule(c3, c3)

        self.ir_down2     = Down(c2, c3)
        self.ir_incep3    = InceptionModule(c3, c3)

        # scale4 (H/8 x W/8)   -- Down3  （显式保留）
        self.rgb_down3    = Down(c3, c4)
        self.rgb_incep4   = InceptionModule(c4, c4)

        self.ms_down3     = Down(c3, c4)
        self.ms_incep4    = InceptionModule(c4, c4)

        self.ir_down3     = Down(c3, c4)
        self.ir_incep4    = InceptionModule(c4, c4)

        # scale5 (H/16 x W/16) -- Down4 （最底层）
        self.rgb_down4    = Down(c4, c5)
        self.rgb_incep5   = InceptionModule(c5, c5)

        self.ms_down4     = Down(c4, c5)
        self.ms_incep5    = InceptionModule(c5, c5)

        self.ir_down4     = Down(c4, c5)
        self.ir_incep5    = InceptionModule(c5, c5)

        # bottom merge conv (将 3*c5 -> c5) 已注册为 module
        self.bottom_merge_conv = nn.Conv2d(3 * c5, c5, kernel_size=1, bias=False)

        # 合并输入 9ch 支持（可选）
        if not use_separate_img:
            self.img9_in = ConvBlock(9, c1)

        # --- DenseCrossModal per scale (image-only dense mixing)
        # apply DenseCrossModal at each scale (scale1..scale5)
        self.dense_cross_per_scale = nn.ModuleList([
            DenseCrossModal(in_chs=[c1, c1, c1]),  # scale1 (H)
            DenseCrossModal(in_chs=[c2, c2, c2]),  # scale2 (H/2)
            DenseCrossModal(in_chs=[c3, c3, c3]),  # scale3 (H/4)
            DenseCrossModal(in_chs=[c4, c4, c4]),  # scale4 (H/8)
            DenseCrossModal(in_chs=[c5, c5, c5])   # scale5 (H/16)
        ])

        # --- CrossModalGating per scale (fuse image modalities only)
        self.gates = nn.ModuleList([
            CrossModalGating(in_chs=[c1, c1, c1], out_ch=c1),  # f1
            CrossModalGating(in_chs=[c2, c2, c2], out_ch=c2),  # f2
            CrossModalGating(in_chs=[c3, c3, c3], out_ch=c3),  # f3
            CrossModalGating(in_chs=[c4, c4, c4], out_ch=c4),  # f4
            CrossModalGating(in_chs=[c5, c5, c5], out_ch=c5)   # f5 bottom
        ])

        # --- TabularProjector: MTF(10) + SHAP(pos+neg+rank 3*10=30) -> emb_dim (choose c5)
        self.tab_projector = TabularProjector(mtf_dim, shap_dim=mtf_dim*3, emb_dim=c5)

        # --- bottom reduce + inception + bottleneck
        # bottom_reduce: 将 concat(bottom_img_features, tab_map) 的通道压回 c5
        self.bottom_reduce = nn.Conv2d(c5 * 2, c5, kernel_size=1)  # note: we will concat bottom (c5) and tab_map (c5) -> 2*c5
        self.bottom_inception = InceptionModule(c5, c5)
        self.bottleneck = ConvBlock(c5, c5)

        # --- decoder ups (clean channel signature), 对称4次上采样
        # 输入 bott (c5), skip from f5 (c5) -> out c4
        self.up4 = Up(up_in_ch=c5, skip_ch=c5, out_ch=c4)
        # up3: in c4, skip c4 -> out c3
        self.up3 = Up(up_in_ch=c4, skip_ch=c4, out_ch=c3)
        # up2: in c3, skip c3 -> out c2
        self.up2 = Up(up_in_ch=c3, skip_ch=c3, out_ch=c2)
        # up1: in c2, skip c2 -> out c1
        self.up1 = Up(up_in_ch=c2, skip_ch=c2, out_ch=c1)

        # final conv: concat(d1, f1) -> reduce to c1
        self.final_conv = ConvBlock(c1 + c1, c1)

        # reg head
        self.reg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, max(32, c1//1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(32, c1//1), 1)
        )

    def forward(self, img_input, ms_input=None, ir_input=None,
                excel=None, shap_pos=None, shap_neg=None, shap_rank=None):
        """
        img_input: if use_separate_img True => rgb tensor [B,3,H,W]
                   else => img_input [B,9,H,W] (3+5+1)
        ms_input: [B,5,H,W]
        ir_input: [B,1,H,W]
        excel: [B,10]
        shap_pos/shap_neg/shap_rank: [10] or [B,10] or None
        """
        B = img_input.shape[0]
        # prepare rgb/ms/ir
        if self.use_separate_img:
            rgb = img_input
            ms  = ms_input
            ir  = ir_input
        else:
            rgb = img_input[:, :3, :, :]
            ms  = img_input[:, 3:8, :, :]
            ir  = img_input[:, 8:, :, :]

        H0, W0 = rgb.shape[2], rgb.shape[3]
        spatial_sizes = [
            (H0, W0),           # scale1
            (H0//2, W0//2),     # scale2
            (H0//4, W0//4),     # scale3
            (H0//8, W0//8),     # scale4
            (H0//16, W0//16)    # scale5 (bottom)
        ]

        # -------------------
        # encoder per-modality + local Inception refine
        # -------------------
        # scale1 (H x W)
        r1 = self.rgb_in(rgb); r1 = self.rgb_incep1(r1)
        m1 = self.ms_in(ms);  m1 = self.ms_incep1(m1)
        ir1 = self.ir_in(ir); ir1 = self.ir_incep1(ir1)

        # down1 -> scale2 (H/2)
        r_pool1, r_skip1 = self.rgb_down1(r1); r_skip1 = self.rgb_incep2(r_skip1)
        m_pool1, m_skip1 = self.ms_down1(m1);  m_skip1 = self.ms_incep2(m_skip1)
        ir_pool1, ir_skip1 = self.ir_down1(ir1); ir_skip1 = self.ir_incep2(ir_skip1)

        # down2 -> scale3 (H/4)
        r_pool2, r_skip2 = self.rgb_down2(r_pool1); r_skip2 = self.rgb_incep3(r_skip2)
        m_pool2, m_skip2 = self.ms_down2(m_pool1);  m_skip2 = self.ms_incep3(m_skip2)
        ir_pool2, ir_skip2 = self.ir_down2(ir_pool1); ir_skip2 = self.ir_incep3(ir_skip2)

        # down3 -> scale4 (H/8)
        r_pool3, r_skip3 = self.rgb_down3(r_pool2); r_skip3 = self.rgb_incep4(r_skip3)
        m_pool3, m_skip3 = self.ms_down3(m_pool2);  m_skip3 = self.ms_incep4(m_skip3)
        ir_pool3, ir_skip3 = self.ir_down3(ir_pool2); ir_skip3 = self.ir_incep4(ir_skip3)

        # down4 -> scale5 (H/16) bottom pool
        r_pool4, r_skip4 = self.rgb_down4(r_pool3); r_skip4 = self.rgb_incep5(r_skip4)
        m_pool4, m_skip4 = self.ms_down4(m_pool3);  m_skip4 = self.ms_incep5(m_skip4)
        ir_pool4, ir_skip4 = self.ir_down4(ir_pool3); ir_skip4 = self.ir_incep5(ir_skip4)

        # -------------------
        # DenseCrossModal per scale (image-only dense mixing)
        # 每个尺度都使用 DenseCrossModal -> 产生每个模态增强后的同尺度特征
        # -------------------
        # scale1 dense (full res)
        r1_e, m1_e, ir1_e = self.dense_cross_per_scale[0]([r1, m1, ir1])
        # scale2 dense (H/2)
        r2_e, m2_e, ir2_e = self.dense_cross_per_scale[1]([r_skip1, m_skip1, ir_skip1])
        # scale3 dense (H/4)
        r3_e, m3_e, ir3_e = self.dense_cross_per_scale[2]([r_skip2, m_skip2, ir_skip2])
        # scale4 dense (H/8)
        r4_e, m4_e, ir4_e = self.dense_cross_per_scale[3]([r_skip3, m_skip3, ir_skip3])
        # scale5 dense (H/16)
        r5_e, m5_e, ir5_e = self.dense_cross_per_scale[4]([r_skip4, m_skip4, ir_skip4])

        # -------------------
        # CrossModalGating per scale (fuse image-only features)
        # 每个尺度都使用 gating + Inception fuse -> 得到 fused skip features f1..f5
        # -------------------
        f1, g1 = self.gates[0]([r1_e, m1_e, ir1_e])   # f1: [B, c1, H, W]
        f2, g2 = self.gates[1]([r2_e, m2_e, ir2_e])   # f2: [B, c2, H/2, W/2]
        f3, g3 = self.gates[2]([r3_e, m3_e, ir3_e])   # f3: [B, c3, H/4, W/4]
        f4, g4 = self.gates[3]([r4_e, m4_e, ir4_e])   # f4: [B, c4, H/8, W/8]
        f5, g5 = self.gates[4]([r5_e, m5_e, ir5_e])   # f5: [B, c5, H/16, W/16]

        # -------------------
        # Tabular (MTF + SHAP pos/neg/rank) -> embedding broadcast to bottom spatial size
        # -------------------
        hb, wb = spatial_sizes[4]  # bottom H,W = H/16, W/16
        tab_map_b, shap_gate_vec = self.tab_projector(excel,
                                                      shap_pos=shap_pos,
                                                      shap_neg=shap_neg,
                                                      shap_rank=shap_rank,
                                                      target_hw=(hb, wb))  # [B, c5, hb, wb], [B, emb_dim]

        # -------------------
        # bottom: concat pooled image modalities (r_pool4, m_pool4, ir_pool4)
        # then reduce -> Inception -> concat with tab_map_b -> merge conv -> bottleneck
        # -------------------
        bottom_cat = torch.cat([r_pool4, m_pool4, ir_pool4], dim=1)  # [B, c5*3, hb, wb]
        bottom = self.bottom_merge_conv(bottom_cat)  # [B, c5, hb, wb]
        bottom = self.bottom_inception(bottom)   # Inception refine at bottom (c5)

        # concat with tab_map_b (text+shap guidance)
        bottom = torch.cat([bottom, tab_map_b], dim=1)  # [B, 2*c5, hb, wb]
        bottom = self.bottom_reduce(bottom)             # [B, c5, hb, wb]
        bott = self.bottleneck(bottom)                  # [B, c5, hb, wb]

        # -------------------
        # decoder uses fused skip features f5..f1 (注意 f5 is used as skip for up4)
        # -------------------
        d4 = self.up4(bott, f5)  # -> out c4 (H/8)
        d3 = self.up3(d4, f4)    # -> out c3 (H/4)
        d2 = self.up2(d3, f3)    # -> out c2 (H/2)
        d1 = self.up1(d2, f2)    # -> out c1 (H)

        final = torch.cat([d1, f1], dim=1)  # concat with top-level fused skip (f1)
        final = self.final_conv(final)      # [B, c1, H, W]

        out = self.reg_head(final)  # [B,1]

        return out.squeeze(1), {
            "shap_gate": shap_gate_vec,       # [B, emb_dim] (注意：emb_dim == c5)
            "gates": [g1, g2, g3, g4, g5],    # per-scale gating maps
            "fused_skips": [f1, f2, f3, f4, f5]
        }