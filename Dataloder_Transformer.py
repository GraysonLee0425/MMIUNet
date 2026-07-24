# dataloader.py
# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import tifffile as tiff


class MyDataset(Dataset):
    """
    输入：
        RGB: 3通道
        MS : 5通道
        TIR: 1通道
        MulD: Excel筛选10维特征

    输出：
        rgb : [3,256,256]
        ms  : [5,256,256]
        tir : [1,256,256]
        muld: [10]
        y   : 标量
    """

    def __init__(self,
                 img_path_rgb,
                 img_path_ms1, img_path_ms2, img_path_ms3, img_path_ms4, img_path_ms5,
                 img_path_tir,
                 label_path,
                 excel_path,
                 sheet_name_excel="开花期",
                 target_col="Y_Y13",
                 img_size=256):

        super().__init__()

        # ===== 路径 =====
        self.img_path_rgb = img_path_rgb
        self.ms_paths = [img_path_ms1, img_path_ms2, img_path_ms3,
                         img_path_ms4, img_path_ms5]
        self.img_path_tir = img_path_tir

        self.label_path = label_path
        self.excel_path = excel_path
        self.sheet_name_excel = sheet_name_excel
        self.target_col = target_col
        self.img_size = img_size

        # ===== 标签 =====
        with open(label_path, 'r', encoding='utf-8') as f:
            self.anno = json.load(f)

        # ===== MulD特征（10维）=====
        excel_df = pd.read_excel(excel_path, sheet_name=sheet_name_excel)
        self.muld_features = excel_df.iloc[:, 0:10].values.astype(np.float32)

        # 防止样本数不一致
        if len(self.muld_features) < len(self.anno):
            rep = int(np.ceil(len(self.anno) / len(self.muld_features)))
            self.muld_features = np.tile(self.muld_features, (rep, 1))
        self.muld_features = self.muld_features[:len(self.anno)]

        # ===== 变换 =====
        self.resize = transforms.Resize((img_size, img_size))
        self.to_tensor = transforms.ToTensor()

        # RGB均值（建议用你原始统计值）
        self.rgb_norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.anno)

    # --------------------------------------------------
    # 工具函数
    # --------------------------------------------------
    def open_image(self, file_path):
        if file_path.lower().endswith(".tif"):
            with tiff.TiffFile(file_path) as tif:
                return tif.asarray()
        else:
            with Image.open(file_path) as img:
                return np.array(img)

    def load_single_channel(self, path, size):
        img = self.open_image(path)
        if img.ndim == 3:
            img = img[:, :, 0]
        img = Image.fromarray(img).convert('L')
        img = img.resize(size, Image.BILINEAR)
        return self.to_tensor(img)  # 1,H,W

    # --------------------------------------------------
    # 主函数
    # --------------------------------------------------
    def __getitem__(self, idx):
        a = self.anno[idx]
        image_name = a['image_name']

        # ===== RGB =====
        rgb_path = self.img_path_rgb + image_name
        rgb_img = Image.open(rgb_path).convert('RGB')
        rgb_img = self.resize(rgb_img)
        rgb = self.to_tensor(rgb_img)
        rgb = self.rgb_norm(rgb)   # [3,256,256]

        target_size = rgb_img.size

        # ===== MS (5通道) =====
        ms_list = []
        for p in self.ms_paths:
            path = p + image_name
            ms_list.append(self.load_single_channel(path, target_size))
        ms = torch.cat(ms_list, dim=0)   # [5,256,256]

        # ===== TIR =====
        tir_path = self.img_path_tir + image_name
        tir = self.load_single_channel(tir_path, target_size)  # [1,256,256]

        # ===== MulD =====
        muld = torch.tensor(self.muld_features[idx], dtype=torch.float32)  # [10]

        # ===== Label =====
        y = torch.tensor(a[self.target_col], dtype=torch.float32)

        return ms, rgb, tir, muld, y