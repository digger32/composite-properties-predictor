"""
train_models.py — обучение и сохранение итоговых моделей ВКР.

Модели сохраняются в директорию models/:
    ridge_modulus.joblib        — Ridge-регрессия для модуля упругости при растяжении
    rf_strength.joblib          — Random Forest для прочности при растяжении
    scaler_23.joblib            — StandardScaler для входов моделей раздела 2.3
    mlp_ratio.pt                — state_dict нейронной сети (PyTorch)
    scaler_mlp_X.joblib         — StandardScaler для входов нейронной сети (раздел 2.4)
    scaler_mlp_y.joblib         — StandardScaler для выхода нейронной сети
    metadata.json               — имена признаков, метрики, версия
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

TARGET_MODULUS = "Модуль упругости при растяжении, ГПа"
TARGET_STRENGTH = "Прочность при растяжении, МПа"
TARGET_RATIO = "Соотношение матрица-наполнитель"

# Порядок признаков фиксируется явно, чтобы приложение и обучение были синхронны
FEATURES_23 = [
    "Соотношение матрица-наполнитель",
    "Плотность, кг/м3",
    "модуль упругости, ГПа",
    "Количество отвердителя, м.%",
    "Содержание эпоксидных групп,%_2",
    "Температура вспышки, С_2",
    "Поверхностная плотность, г/м2",
    "Потребление смолы, г/м2",
    "Угол нашивки, град",
    "Шаг нашивки",
    "Плотность нашивки",
]
FEATURES_24 = [f for f in FEATURES_23 if f != TARGET_RATIO]


class MLPRatio(nn.Module):
    """Архитектура MLP-2.4."""

    def __init__(self, n_in: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_dataset() -> pd.DataFrame:
    """Загружает и объединяет два Excel-файла датасета по индексу (INNER JOIN)."""
    df_bp = pd.read_excel(DATA_DIR / "X_bp.xlsx", index_col=0)
    df_nup = pd.read_excel(DATA_DIR / "X_nup.xlsx", index_col=0)
    df = df_bp.join(df_nup, how="inner")
    if df.isna().any().any() or df.duplicated().any():
        raise ValueError("Датасет содержит пропуски или дубликаты")
    return df


def train_regression_models(df: pd.DataFrame) -> dict[str, dict]:
    """Обучает Ridge (для модуля) и RandomForest (для прочности) и возвращает метрики."""
    X = df[FEATURES_23].values
    y_mod = df[TARGET_MODULUS].values
    y_str = df[TARGET_STRENGTH].values

    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)

    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)

    # Ridge для модуля упругости (минимум переобучения, ближе всего к baseline)
    Xtr, Xte, ytr, yte = train_test_split(X_s, y_mod, test_size=0.3, random_state=SEED)
    ridge_mod = Ridge(alpha=10.0, random_state=SEED).fit(Xtr, ytr)
    metrics_mod = _metrics(ridge_mod, X_s, y_mod, Xtr, ytr, Xte, yte, cv)

    # Random Forest для прочности (единственная модель, превосходящая baseline по R²(CV))
    Xtr, Xte, ytr, yte = train_test_split(X_s, y_str, test_size=0.3, random_state=SEED)
    rf_str = RandomForestRegressor(
        n_estimators=200, max_depth=5, min_samples_leaf=5,
        random_state=SEED, n_jobs=-1,
    ).fit(Xtr, ytr)
    metrics_str = _metrics(rf_str, X_s, y_str, Xtr, ytr, Xte, yte, cv)

    joblib.dump(ridge_mod, MODELS_DIR / "ridge_modulus.joblib")
    joblib.dump(rf_str, MODELS_DIR / "rf_strength.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler_23.joblib")

    return {"modulus": metrics_mod, "strength": metrics_str}


def _metrics(model, X_full, y_full, Xtr, ytr, Xte, yte, cv) -> dict:
    ptr, pte = model.predict(Xtr), model.predict(Xte)
    cv_r2 = cross_val_score(model, X_full, y_full, cv=cv, scoring="r2", n_jobs=-1).mean()
    return {
        "MAE_test": round(mean_absolute_error(yte, pte), 4),
        "RMSE_test": round(float(np.sqrt(mean_squared_error(yte, pte))), 4),
        "R2_test": round(r2_score(yte, pte), 4),
        "R2_CV": round(float(cv_r2), 4),
        "R2_train": round(r2_score(ytr, ptr), 4),
    }


def train_mlp(df: pd.DataFrame) -> dict:
    """Обучает MLP для рекомендации соотношения матрица-наполнитель."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X = df[FEATURES_24].values.astype(np.float32)
    y = df[TARGET_RATIO].values.astype(np.float32)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED)
    Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.15, random_state=SEED)

    x_scaler = StandardScaler().fit(Xtr)
    y_scaler = StandardScaler().fit(ytr.reshape(-1, 1))
    Xtr_s = x_scaler.transform(Xtr).astype(np.float32)
    Xval_s = x_scaler.transform(Xval).astype(np.float32)
    Xte_s = x_scaler.transform(Xte).astype(np.float32)
    ytr_s = y_scaler.transform(ytr.reshape(-1, 1)).astype(np.float32).ravel()
    yval_s = y_scaler.transform(yval.reshape(-1, 1)).astype(np.float32).ravel()

    model = MLPRatio(n_in=len(FEATURES_24))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr_s), torch.tensor(ytr_s).unsqueeze(1)),
        batch_size=32, shuffle=True,
    )
    Xval_t, yval_t = torch.tensor(Xval_s), torch.tensor(yval_s).unsqueeze(1)

    best_val, patience_cnt, best_state = float("inf"), 0, None
    for _ in range(300):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            nn.functional.mse_loss(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(model(Xval_t), yval_t).item()
        scheduler.step(val_loss)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 30:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_s = model(torch.tensor(Xte_s)).numpy().ravel()
    pred = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()

    torch.save(model.state_dict(), MODELS_DIR / "mlp_ratio.pt")
    joblib.dump(x_scaler, MODELS_DIR / "scaler_mlp_X.joblib")
    joblib.dump(y_scaler, MODELS_DIR / "scaler_mlp_y.joblib")

    return {
        "MAE_test": round(mean_absolute_error(yte, pred), 4),
        "RMSE_test": round(float(np.sqrt(mean_squared_error(yte, pred))), 4),
        "R2_test": round(r2_score(yte, pred), 4),
    }


def main() -> None:
    print("Загрузка датасета …")
    df = load_dataset()
    print(f"  размер: {df.shape[0]} строк × {df.shape[1]} столбцов\n")

    print("Обучение регрессионных моделей (Ridge, Random Forest) …")
    reg_metrics = train_regression_models(df)
    for target, m in reg_metrics.items():
        print(f"  {target}: {m}")

    print("\nОбучение нейронной сети MLP для соотношения матрица-наполнитель …")
    mlp_metrics = train_mlp(df)
    print(f"  ratio: {mlp_metrics}")

    metadata = {
        "version": "1.0",
        "features_23": FEATURES_23,
        "features_24": FEATURES_24,
        "target_modulus": TARGET_MODULUS,
        "target_strength": TARGET_STRENGTH,
        "target_ratio": TARGET_RATIO,
        "metrics": {**reg_metrics, "ratio": mlp_metrics},
        "seed": SEED,
    }
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\n✓ Все модели сохранены в {MODELS_DIR.resolve()}")


if __name__ == "__main__":
    main()
