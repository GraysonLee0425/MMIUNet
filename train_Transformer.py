# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from tqdm import tqdm

from Dataloder_Transformer import MyDataset
from model_Transformer import MultiModalDecisionViT


def train_main(args):

    rgb_path = "F:/wq-data/2023seg/rgb/"

    ms1_path = "F:/wq-data/2023seg/ms/Blue/"
    ms2_path = "F:/wq-data/2023seg/ms/Green/"
    ms3_path = "F:/wq-data/2023seg/ms/Red/"
    ms4_path = "F:/wq-data/2023seg/ms/RedEdge/"
    ms5_path = "F:/wq-data/2023seg/ms/NIR/"

    tir_path = "F:/wq-data/2023seg/tir/"

    label_path_train = "F:/wq-data/json/train.json"
    label_path_val   = "F:/wq-data/json/val.json"

    excel_path = "F:/wq-data/feature/MulD.xlsx"
    sheet_name_phase = "开花期"

    save_dir = "DecisionViT_Result"
    os.makedirs(save_dir, exist_ok=True)

    # =========================
    # Dataset
    # =========================
    TrainDataset = MyDataset(
        rgb_path,
        ms1_path, ms2_path, ms3_path, ms4_path, ms5_path,
        tir_path,
        label_path_train,
        excel_path,
        sheet_name_excel=sheet_name_phase
    )

    ValDataset = MyDataset(
        rgb_path,
        ms1_path, ms2_path, ms3_path, ms4_path, ms5_path,
        tir_path,
        label_path_val,
        excel_path,
        sheet_name_excel=sheet_name_phase
    )

    train_loader = torch.utils.data.DataLoader(
        TrainDataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = torch.utils.data.DataLoader(
        ValDataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # =========================
    # 模型
    # =========================
    model = MultiModalDecisionViT(
        img_size=256,
        patch_size=16,
        embed_dim=768,
        depth=12,
        decision_depth=2,
        muld_dim=10
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    # =========================
    # 训练循环
    # =========================
    n_epochs = args.epochs
    logs = []

    best_val_r2 = -np.inf
    best_epoch = -1

    pbar = tqdm(range(n_epochs), unit="epoch")

    for ep in pbar:

        # ===== Train =====
        model.train()
        running_loss = 0.0
        y_trues, y_preds = [], []

        for batch in train_loader:

            ms, rgb, tir, muld, y = batch

            ms = ms.to(device)
            rgb = rgb.to(device)
            tir = tir.to(device)
            muld = muld.to(device)
            y = y.to(device).unsqueeze(1)

            optimizer.zero_grad()

            y_pred = model(ms, rgb, tir, muld)

            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            y_trues.extend(y.detach().cpu().numpy())
            y_preds.extend(y_pred.detach().cpu().numpy())

        # ===== Train指标 =====
        train_r2 = metrics.r2_score(y_trues, y_preds)
        train_mae = metrics.mean_absolute_error(y_trues, y_preds)
        train_rmse = np.sqrt(metrics.mean_squared_error(y_trues, y_preds))

        # ===== Validation =====
        model.eval()
        val_trues, val_preds = [], []

        with torch.no_grad():
            for batch in val_loader:
                ms, rgb, tir, muld, y = batch

                ms = ms.to(device)
                rgb = rgb.to(device)
                tir = tir.to(device)
                muld = muld.to(device)
                y = y.to(device).unsqueeze(1)

                y_pred = model(ms, rgb, tir, muld)

                val_trues.extend(y.cpu().numpy())
                val_preds.extend(y_pred.cpu().numpy())

        val_r2 = metrics.r2_score(val_trues, val_preds)
        val_mae = metrics.mean_absolute_error(val_trues, val_preds)
        val_rmse = np.sqrt(metrics.mean_squared_error(val_trues, val_preds))

        pbar.set_postfix({
            "TrainR2": f"{train_r2:.3f}",
            "ValR2": f"{val_r2:.3f}"
        })

        logs.append([
            ep,
            running_loss / max(1, len(train_loader)),
            train_r2, val_r2,
            train_mae, val_mae,
            train_rmse, val_rmse
        ])

        # ===== 保存最佳模型 =====
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = ep

            torch.save(model.state_dict(),
                       os.path.join(save_dir, "best_model.pth"))

    # 保存日志
    pd.DataFrame(logs, columns=[
        'Epoch', 'Train Loss',
        'Train R2', 'Val R2',
        'Train MAE', 'Val MAE',
        'Train RMSE', 'Val RMSE'
    ]).to_excel(os.path.join(save_dir, "training_log.xlsx"), index=False)

    print("Best Epoch:", best_epoch)
    print("Best Val R2:", best_val_r2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    train_main(args)