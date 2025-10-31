# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
from Dataloder_Unet import MyDataset
from model_Unet import MultiModalUNet  # 你前面写的Unet融合模型

def train_main(args):
    # ===== 路径（按你本地改）=====
    rgb_path = "F:/wq-data/2023seg/2023_mul/DEseg_2/rgb/"
    ms_path  = "F:/wq-data/2023seg/2023_mul/DEseg_2/multispectral/"  # 假设你合成了5通道文件
    label_path_train = 'F:/wq-data/2023seg/2023_mul/DE/DE_JSON/2/2_train.json'
    label_path_val   = 'F:/wq-data/2023seg/2023_mul/DE/DE_JSON/2/2_val.json'
    excel_path       = 'F:/wq-data/2023seg/2023_mul/DE/DE_Fea.xlsx'
    sheet_name_phase = "开花期"

    save_dir = "UNetFusion_Result"
    os.makedirs(save_dir, exist_ok=True)

    # ===== Dataset / DataLoader =====
    TrainDataset = MyDataset(rgb_path, ms_path, label_path_train, excel_path,
                             sheet_name_excel=sheet_name_phase)
    ValDataset   = MyDataset(rgb_path, ms_path, label_path_val, excel_path,
                             sheet_name_excel=sheet_name_phase)

    train_loader = torch.utils.data.DataLoader(TrainDataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = torch.utils.data.DataLoader(ValDataset,   batch_size=16, shuffle=False, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ===== 模型 =====
    model = MultiModalUNet(text_dim=10).to(device)  # 10维文本特征

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    # ===== 训练循环 =====
    n_epochs = args.epochs
    logs = []

    best_val_r2 = -np.inf
    best_epoch = -1
    best_metrics = None

    pbar = tqdm(range(n_epochs), unit="epoch")
    for ep in pbar:
        model.train()
        running_loss = 0.0
        y_trues, y_preds = [], []

        for batch in train_loader:
            rgb, ms, excel, shap_pos, shap_neg, y = batch
            rgb, ms = rgb.to(device), ms.to(device)
            excel, shap_pos, shap_neg = excel.to(device), shap_pos.to(device), shap_neg.to(device)
            y = y.to(device).unsqueeze(1)

            optimizer.zero_grad()
            y_pred, aux = model(rgb, ms, excel, shap_pos, shap_neg)  # <<< 改这里
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            y_trues.extend(y.detach().cpu().numpy())
            y_preds.extend(y_pred.detach().cpu().numpy())

        # ===== 训练指标 =====
        train_r2 = metrics.r2_score(y_trues, y_preds)
        train_mae = metrics.mean_absolute_error(y_trues, y_preds)
        train_rmse = np.sqrt(metrics.mean_squared_error(y_trues, y_preds))

        # ===== 验证 =====
        model.eval()
        val_trues, val_preds = [], []
        with torch.no_grad():
            for batch in val_loader:
                rgb, ms, excel, shap_pos, shap_neg, y = batch
                rgb, ms = rgb.to(device), ms.to(device)
                excel, shap_pos, shap_neg = excel.to(device), shap_pos.to(device), shap_neg.to(device)
                y = y.to(device).unsqueeze(1)

                y_pred, aux = model(rgb, ms, excel, shap_pos, shap_neg)  # <<< 改这里
                val_trues.extend(y.cpu().numpy())
                val_preds.extend(y_pred.cpu().numpy())

        val_r2 = metrics.r2_score(val_trues, val_preds)
        val_mae = metrics.mean_absolute_error(val_trues, val_preds)
        val_rmse = np.sqrt(metrics.mean_squared_error(val_trues, val_preds))

        # tqdm 显示
        pbar.set_postfix({
            "TrainR2": f"{train_r2:.3f}",
            "ValR2": f"{val_r2:.3f}"
        })

        # 日志
        logs.append([ep, running_loss / max(1, len(train_loader)),
                     train_r2, val_r2,
                     train_mae, val_mae,
                     train_rmse, val_rmse])

        # ===== 保存最优 =====
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = ep
            best_metrics = logs[-1]

            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            pd.DataFrame([best_metrics], columns=[
                'Epoch', 'Train Loss',
                'Train R2', 'Val R2',
                'Train MAE', 'Val MAE',
                'Train RMSE', 'Val RMSE'
            ]).to_excel(os.path.join(save_dir, "best_metrics.xlsx"), index=False)

    # 训练结束保存完整日志
    pd.DataFrame(logs, columns=[
        'Epoch', 'Train Loss',
        'Train R2', 'Val R2',
        'Train MAE', 'Val MAE',
        'Train RMSE', 'Val RMSE'
    ]).to_excel(os.path.join(save_dir, "training_log.xlsx"), index=False)

    print(f"最佳模型出现在 Epoch {best_epoch}，Val R² = {best_val_r2:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()
    train_main(args)
