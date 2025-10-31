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
      - 图像: RGB(3通道) + 多光谱(5通道) = 8通道
      - Excel 特征: 10维
      - SHAP: 按 Excel 特征对齐，分为正/负向，单任务时只取一个目标的列
    输出：
      img_8c: [8,H,W]
      excel_feature: [10]
      shap_pos: [10]
      shap_neg: [10]
      y: 标量 (产量单任务回归)
    """
    def __init__(self,
                 img_path0, img_path1, img_path2, img_path3, img_path4, img_path5,
                 label_path,
                 excel_path,
                 sheet_name_excel="开花期",
                 shap_path=None,
                 sheet_name_shap="开花期",
                 target_col="Y_Y13"):   # 👈 这里指定你要预测的单任务列
        super().__init__()
        self.img_path0 = img_path0
        self.img_path1 = img_path1
        self.img_path2 = img_path2
        self.img_path3 = img_path3
        self.img_path4 = img_path4
        self.img_path5 = img_path5
        self.label_path = label_path
        self.excel_path = excel_path
        self.sheet_name_excel = sheet_name_excel
        self.shap_path = shap_path
        self.sheet_name_shap = sheet_name_shap
        self.target_col = target_col

        # ====== 载入标签 ======
        with open(label_path, 'r', encoding='utf-8') as f:
            self.anno = json.load(f)

        # ====== Excel 特征 (10维) ======
        excel_df = pd.read_excel(excel_path, sheet_name=sheet_name_excel)
        self.excel_feature_names = list(excel_df.columns[:10])
        self.excel_features = excel_df.iloc[:, 0:10].values.astype(np.float32)

        if len(self.excel_features) < len(self.anno):
            rep = int(np.ceil(len(self.anno) / len(self.excel_features)))
            self.excel_features = np.tile(self.excel_features, (rep, 1))
        self.excel_features = self.excel_features[:len(self.anno)]

        # ====== SHAP 表 (取单任务正/负向) ======
        if shap_path is not None:
            shap_df = pd.read_excel(shap_path, sheet_name=sheet_name_shap)
            shap_feat_col = shap_df.columns[0]
            shap_df = shap_df.set_index(shap_feat_col)

            pos_col = f"{target_col}_pos"
            neg_col = f"{target_col}_neg"
            if pos_col not in shap_df.columns or neg_col not in shap_df.columns:
                raise ValueError(f"SHAP表缺少 {pos_col}/{neg_col}")

            shap_pos, shap_neg = [], []
            for fname in self.excel_feature_names:
                row = shap_df.loc[fname]
                shap_pos.append(row[pos_col])
                shap_neg.append(row[neg_col])

            self.shap_pos = np.array(shap_pos, dtype=np.float32)  # [10]
            self.shap_neg = np.array(shap_neg, dtype=np.float32)  # [10]
        else:
            self.shap_pos = np.zeros((10,), dtype=np.float32)
            self.shap_neg = np.zeros((10,), dtype=np.float32)

        # ====== 图像增强 ======
        self.normalize = transforms.Normalize(
            [92.48556564, 95.25143508, 61.86386833,
             0.027386447, 0.03902037, 0.025460988,
             0.137763652, 0.33754906],
            [36.9384424, 36.04451126, 27.76898411,
             0.014065378, 0.015877601, 0.009822546,
             0.048675962, 0.113302902]
        )
        self.rand_crop = transforms.RandomResizedCrop(224)

    def __len__(self):
        return len(self.anno)

    def open_image(self, file_path, use_tifffile=False):
        if use_tifffile:
            with tiff.TiffFile(file_path) as tif:
                return tif.asarray()
        else:
            with Image.open(file_path) as img:
                return np.array(img)

    def _load_single_channel(self, path, target_size):
        img = self.open_image(path)
        if img.ndim == 3:
            img = img[:, :, 0]
        img_pil = Image.fromarray(img).convert('L')
        img_pil = img_pil.resize(target_size, Image.BILINEAR)
        return np.array(img_pil)

    def __getitem__(self, idx):
        a = self.anno[idx]
        image_name = a['image_name']

        # ====== 图像路径 ======
        p0 = self.img_path0 + image_name
        p1 = self.img_path1 + image_name
        p2 = self.img_path2 + image_name
        p3 = self.img_path3 + image_name
        p4 = self.img_path4 + image_name
        p5 = self.img_path5 + image_name

        # RGB 尺寸基准
        img0 = self.open_image(p0)
        img0_pil = Image.fromarray(img0).convert('RGB')
        target_size = img0_pil.size

        sc = lambda path: self._load_single_channel(path, target_size)
        img1 = sc(p1); img2 = sc(p2); img3 = sc(p3); img4 = sc(p4); img5 = sc(p5)

        # ====== 拼接 8通道 ======
        img0_tensor = transforms.ToTensor()(img0_pil)     # (3,H,W)
        stacked = np.stack([img1, img2, img3, img4, img5], axis=-1)  # (H,W,5)
        stacked_tensor = transforms.ToTensor()(stacked)   # (5,H,W)
        img_8c = torch.cat((img0_tensor, stacked_tensor), dim=0)  # (8,H,W)
        img_8c = self.normalize(img_8c)
        img_8c = self.rand_crop(img_8c).float()

        # ====== Excel特征 ======
        excel_feature = torch.tensor(self.excel_features[idx], dtype=torch.float32)  # [10]

        # ====== SHAP 特征（正负双分支） ======
        shap_pos = torch.tensor(self.shap_pos, dtype=torch.float32)  # [10]
        shap_neg = torch.tensor(self.shap_neg, dtype=torch.float32)  # [10]

        # ====== 单任务标签 ======
        y = torch.tensor(a[self.target_col], dtype=torch.float32)  # 👈 只取一个目标

        return img_8c, excel_feature, shap_pos, shap_neg, y
