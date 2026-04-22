"""train_models.py — воспроизводимый пайплайн обучения всех моделей ВКР.

Методика:
1. Исходные таблицы X_bp.xlsx и X_nup.xlsx объединяются INNER JOIN по индексу.
2. Признаки X стандартизуются StandardScaler (требование подраздела 2.1).
3. Целевые переменные Y внутренне стандартизуются StandardScaler для
   обеспечения численной устойчивости градиентных методов (MLP, SVR), но
   метрики MAE, RMSE отчётно рассчитываются в ИСХОДНЫХ физических единицах
   (ГПа, МПа) после обратного масштабирования прогнозов. Метрика R²
   инвариантна к масштабу и одинакова в обоих представлениях.
4. Обучение выполняется с 10-fold KFold (shuffle=True, random_state=42)
   через GridSearchCV (требование подраздела 2.2).
5. Обучаются 6 моделей для подраздела 2.3 × 2 целевые переменные + MLP-2.4
   для подраздела 2.4 (без утечки данных).
6. Сохраняются все обученные модели, скейлеры, метрики, лог и metadata.json.

Запуск:
    python train_models.py
"""
from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
SEED = 42
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
RESULTS_XLSX = Path("model_results.xlsx")
METADATA_JSON = MODELS_DIR / "metadata.json"
LOG_FILE = Path("training_log.txt")

TARGET_MODULUS = "Модуль упругости при растяжении, ГПа"
TARGET_STRENGTH = "Прочность при растяжении, МПа"
TARGET_RATIO = "Соотношение матрица-наполнитель"

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

MODELS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("train_models")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


log = setup_logger()


# --------------------------------------------------------------------------- #
def load_dataset() -> pd.DataFrame:
    """Загружает и объединяет X_bp.xlsx и X_nup.xlsx методом INNER JOIN."""
    bp = DATA_DIR / "X_bp.xlsx"
    nup = DATA_DIR / "X_nup.xlsx"
    merged = DATA_DIR / "merged_composites.csv"

    if bp.exists() and nup.exists():
        log.info("Загрузка исходных Excel-файлов X_bp.xlsx и X_nup.xlsx")
        df_bp = pd.read_excel(bp, index_col=0)
        df_nup = pd.read_excel(nup, index_col=0)
        df = df_bp.join(df_nup, how="inner")
        log.info(f"  X_bp:  {df_bp.shape[0]} × {df_bp.shape[1]}")
        log.info(f"  X_nup: {df_nup.shape[0]} × {df_nup.shape[1]}")
        log.info(f"  После INNER JOIN: {df.shape[0]} × {df.shape[1]}")
    elif merged.exists():
        log.info(f"Загрузка уже объединённого датасета {merged}")
        df = pd.read_csv(merged)
    else:
        raise FileNotFoundError(
            f"Не найдены исходные данные: ожидается ({bp}, {nup}) или {merged}"
        )

    if int(df.isna().sum().sum()) or int(df.duplicated().sum()):
        raise ValueError("Датасет содержит пропуски или дубликаты")

    missing = set(FEATURES_23 + [TARGET_MODULUS, TARGET_STRENGTH]) - set(df.columns)
    if missing:
        raise ValueError(f"В датасете отсутствуют колонки: {missing}")
    return df


# --------------------------------------------------------------------------- #
def compute_metrics_original_units(
    model, Xtr, ytr, Xte, yte, X_cv, y_cv, cv: KFold,
) -> dict:
    """Расчёт метрик в исходных единицах Y.

    Примечание: для моделей, обёрнутых в TransformedTargetRegressor, атрибут
    `predict` автоматически выполняет обратное масштабирование Y в исходный
    диапазон, поэтому MAE и RMSE получаются в физических единицах (ГПа, МПа).
    """
    p_tr = model.predict(Xtr)
    p_te = model.predict(Xte)
    r2_cv = cross_val_score(model, X_cv, y_cv, cv=cv, scoring="r2", n_jobs=1).mean()
    return {
        "MAE_train":  round(mean_absolute_error(ytr, p_tr), 4),
        "RMSE_train": round(float(np.sqrt(mean_squared_error(ytr, p_tr))), 4),
        "R2_train":   round(r2_score(ytr, p_tr), 4),
        "R2_CV":      round(float(r2_cv), 4),
        "MAE_test":   round(mean_absolute_error(yte, p_te), 4),
        "RMSE_test":  round(float(np.sqrt(mean_squared_error(yte, p_te))), 4),
        "R2_test":    round(r2_score(yte, p_te), 4),
    }


# --------------------------------------------------------------------------- #
def get_model_specs() -> dict:
    """Модели и сетки гиперпараметров для подраздела 2.3.

    Каждая модель, для которой важна нормализация целевой переменной
    (MLPRegressor, SVR), оборачивается в TransformedTargetRegressor с
    StandardScaler. Для остальных моделей обёртка выполняется для
    единообразия — R² инвариантна к масштабу, MAE и RMSE всегда
    возвращаются в исходных единицах.
    """
    return {
        "LinearRegression": (
            TransformedTargetRegressor(
                regressor=LinearRegression(),
                transformer=StandardScaler(),
            ),
            {},
        ),
        "Ridge": (
            TransformedTargetRegressor(
                regressor=Ridge(random_state=SEED),
                transformer=StandardScaler(),
            ),
            {"regressor__alpha": [0.01, 0.1, 1, 10, 100]},
        ),
        "RandomForest": (
            TransformedTargetRegressor(
                regressor=RandomForestRegressor(random_state=SEED, n_jobs=1),
                transformer=StandardScaler(),
            ),
            {
                "regressor__n_estimators": [200],
                "regressor__max_depth": [3, 5, 10],
                "regressor__min_samples_leaf": [1, 5],
            },
        ),
        "XGBoost": (
            TransformedTargetRegressor(
                regressor=XGBRegressor(random_state=SEED, verbosity=0, n_jobs=1),
                transformer=StandardScaler(),
            ),
            {
                "regressor__n_estimators": [200],
                "regressor__max_depth": [3, 5],
                "regressor__learning_rate": [0.05, 0.1],
                "regressor__reg_lambda": [1.0, 10.0],
            },
        ),
        "SVR (RBF)": (
            TransformedTargetRegressor(
                regressor=SVR(kernel="rbf"),
                transformer=StandardScaler(),
            ),
            {
                "regressor__C": [1, 10],
                "regressor__epsilon": [0.05, 0.1],
            },
        ),
        "MLP": (
            TransformedTargetRegressor(
                regressor=MLPRegressor(
                    random_state=SEED, max_iter=500, early_stopping=True,
                ),
                transformer=StandardScaler(),
            ),
            {
                "regressor__hidden_layer_sizes": [(64,), (64, 32)],
                "regressor__alpha": [0.001, 0.01],
            },
        ),
    }


def train_one_target(
    target_name: str, X_s: np.ndarray, y: np.ndarray, cv: KFold,
    save_prefix: str,
) -> tuple[pd.DataFrame, dict]:
    """Обучает 6 моделей + baseline для одной целевой переменной."""
    log.info(f"\n{'=' * 70}\nЦелевая переменная: {target_name}\n{'=' * 70}")

    Xtr, Xte, ytr, yte = train_test_split(X_s, y, test_size=0.3, random_state=SEED)
    log.info(f"Train: {len(Xtr)}, Test: {len(Xte)}")

    results = []
    saved_models = {}

    # Baseline
    log.info("Обучение: Baseline (DummyRegressor, strategy='mean')")
    base = DummyRegressor(strategy="mean").fit(Xtr, ytr)
    m = compute_metrics_original_units(base, Xtr, ytr, Xte, yte, X_s, y, cv)
    m["Model"], m["BestParams"] = "Baseline (mean)", "—"
    results.append(m)
    joblib.dump(base, MODELS_DIR / f"{save_prefix}_baseline.joblib")

    for name, (model, grid) in get_model_specs().items():
        t0 = time.time()
        log.info(f"Обучение: {name}  (сетка: {grid if grid else '—'})")
        if grid:
            gs = GridSearchCV(model, grid, cv=cv, scoring="r2", n_jobs=1)
            gs.fit(Xtr, ytr)
            best, bp = gs.best_estimator_, gs.best_params_
        else:
            model.fit(Xtr, ytr)
            best, bp = model, {}

        m = compute_metrics_original_units(best, Xtr, ytr, Xte, yte, X_s, y, cv)
        m["Model"] = name
        # Очищаем префикс regressor__ из имён параметров для читаемости
        m["BestParams"] = (
            str({k.replace("regressor__", ""): v for k, v in bp.items()})
            if bp else "—"
        )
        results.append(m)

        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        path = MODELS_DIR / f"{save_prefix}_{safe}.joblib"
        joblib.dump(best, path)
        saved_models[name] = path.name

        log.info(f"  R2train={m['R2_train']:+.4f}  R2CV={m['R2_CV']:+.4f}  "
                 f"R2test={m['R2_test']:+.4f}  "
                 f"MAEte={m['MAE_test']:.3f}  ({time.time() - t0:.1f}s)")

    df = pd.DataFrame(results)[["Model", "MAE_train", "RMSE_train", "R2_train",
                                 "R2_CV", "MAE_test", "RMSE_test", "R2_test",
                                 "BestParams"]]
    return df, saved_models


# --------------------------------------------------------------------------- #
class MLPRatio(nn.Module):
    """Архитектура MLP-2.4 (рисунок 14 пояснительной записки)."""

    def __init__(self, n_in: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp24(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Обучает MLP-2.4 для задачи подраздела 2.4."""
    log.info(f"\n{'=' * 70}\nMLP-2.4 (PyTorch): {TARGET_RATIO}\n{'=' * 70}")
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X = df[FEATURES_24].values.astype(np.float32)
    y = df[TARGET_RATIO].values.astype(np.float32)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED)
    Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.15, random_state=SEED)
    log.info(f"Train: {len(Xtr)}, Val: {len(Xval)}, Test: {len(Xte)}")
    log.info("Входы: 10 признаков (исключены модуль и прочность при растяжении)")

    x_sc = StandardScaler().fit(Xtr)
    y_sc = StandardScaler().fit(ytr.reshape(-1, 1))
    Xtr_s = x_sc.transform(Xtr).astype(np.float32)
    Xval_s = x_sc.transform(Xval).astype(np.float32)
    Xte_s = x_sc.transform(Xte).astype(np.float32)
    ytr_s = y_sc.transform(ytr.reshape(-1, 1)).astype(np.float32).ravel()
    yval_s = y_sc.transform(yval.reshape(-1, 1)).astype(np.float32).ravel()

    model = MLPRatio(n_in=len(FEATURES_24))
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr_s), torch.tensor(ytr_s).unsqueeze(1)),
        batch_size=32, shuffle=True,
    )
    Xval_t = torch.tensor(Xval_s)
    yval_t = torch.tensor(yval_s).unsqueeze(1)

    best_val, patience_cnt, best_state, best_epoch = float("inf"), 0, None, 0
    for epoch in range(300):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            nn.functional.mse_loss(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(model(Xval_t), yval_t).item()
        sched.step(val_loss)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 30:
                log.info(f"Ранняя остановка на эпохе {epoch + 1}")
                break
    log.info(f"Лучшая эпоха: {best_epoch}, best val MSE (scaled) = {best_val:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_tr = y_sc.inverse_transform(
            model(torch.tensor(Xtr_s)).numpy().reshape(-1, 1)).ravel()
        pred_te = y_sc.inverse_transform(
            model(torch.tensor(Xte_s)).numpy().reshape(-1, 1)).ravel()

    base_pred_train = np.full_like(ytr, ytr.mean())
    base_pred_test = np.full_like(yte, ytr.mean())
    base_metrics = {
        "Model": "Baseline (mean)",
        "MAE_train":  round(mean_absolute_error(ytr, base_pred_train), 4),
        "RMSE_train": round(float(np.sqrt(mean_squared_error(ytr, base_pred_train))), 4),
        "R2_train":   0.0,
        "MAE_test":   round(mean_absolute_error(yte, base_pred_test), 4),
        "RMSE_test":  round(float(np.sqrt(mean_squared_error(yte, base_pred_test))), 4),
        "R2_test":    round(r2_score(yte, base_pred_test), 4),
    }
    mlp_metrics = {
        "Model": "MLP-2.4 (PyTorch)",
        "MAE_train":  round(mean_absolute_error(ytr, pred_tr), 4),
        "RMSE_train": round(float(np.sqrt(mean_squared_error(ytr, pred_tr))), 4),
        "R2_train":   round(r2_score(ytr, pred_tr), 4),
        "MAE_test":   round(mean_absolute_error(yte, pred_te), 4),
        "RMSE_test":  round(float(np.sqrt(mean_squared_error(yte, pred_te))), 4),
        "R2_test":    round(r2_score(yte, pred_te), 4),
    }
    log.info(f"Baseline: R2(test) = {base_metrics['R2_test']:+.4f}")
    log.info(f"MLP-2.4:  R2(test) = {mlp_metrics['R2_test']:+.4f}")

    torch.save(model.state_dict(), MODELS_DIR / "mlp_ratio.pt")
    joblib.dump(x_sc, MODELS_DIR / "scaler_mlp_X.joblib")
    joblib.dump(y_sc, MODELS_DIR / "scaler_mlp_y.joblib")

    df_res = pd.DataFrame([base_metrics, mlp_metrics])
    info = {
        "best_epoch": best_epoch,
        "best_val_loss_scaled": round(best_val, 6),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "metrics_baseline": base_metrics,
        "metrics_mlp24": mlp_metrics,
    }
    return df_res, info


# --------------------------------------------------------------------------- #
def select_best_by_cv(table: pd.DataFrame) -> str:
    """Выбирает лучшую модель (без baseline) по R² на кросс-валидации."""
    candidates = table[table["Model"] != "Baseline (mean)"]
    return str(candidates.loc[candidates["R2_CV"].idxmax(), "Model"])


def main() -> None:
    t_start = time.time()
    log.info("=" * 70)
    log.info("ВКР: воспроизводимое обучение всех моделей")
    log.info(f"SEED = {SEED}")
    log.info("=" * 70)

    df = load_dataset()

    log.info("\n[ПОДРАЗДЕЛ 2.3] Обучение регрессионных моделей")
    X23 = df[FEATURES_23].values
    scaler_23 = StandardScaler().fit(X23)
    X23_s = scaler_23.transform(X23)
    joblib.dump(scaler_23, MODELS_DIR / "scaler_23.joblib")
    log.info(f"Признаков: {len(FEATURES_23)}. Стандартизатор сохранён.")
    log.info("Стандартизация Y выполняется внутри моделей через "
             "TransformedTargetRegressor; метрики MAE, RMSE возвращаются "
             "в исходных единицах целевой переменной.")

    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)

    table4, saved_mod = train_one_target(
        TARGET_MODULUS, X23_s, df[TARGET_MODULUS].values, cv,
        save_prefix="modulus",
    )
    table5, saved_str = train_one_target(
        TARGET_STRENGTH, X23_s, df[TARGET_STRENGTH].values, cv,
        save_prefix="strength",
    )

    log.info("\n[ПОДРАЗДЕЛ 2.4] Обучение MLP для соотношения матрица-наполнитель")
    table_mlp24, mlp24_info = train_mlp24(df)

    # Выбор лучших моделей и копирование под стабильными именами для приложения
    best_modulus = select_best_by_cv(table4)
    best_strength = select_best_by_cv(table5)
    log.info(f"\nЛучшая модель для модуля упругости (по R²CV): {best_modulus}")
    log.info(f"Лучшая модель для прочности (по R²CV): {best_strength}")

    def _safe(n: str) -> str:
        return n.replace(" ", "_").replace("(", "").replace(")", "")

    import shutil
    shutil.copy(MODELS_DIR / f"modulus_{_safe(best_modulus)}.joblib",
                MODELS_DIR / "best_modulus.joblib")
    shutil.copy(MODELS_DIR / f"strength_{_safe(best_strength)}.joblib",
                MODELS_DIR / "best_strength.joblib")
    log.info("Лучшие модели скопированы как best_modulus.joblib и "
             "best_strength.joblib (используются приложением).")

    with pd.ExcelWriter(RESULTS_XLSX, engine="openpyxl") as writer:
        table4.to_excel(writer, sheet_name="Таблица_4_Модуль", index=False)
        table5.to_excel(writer, sheet_name="Таблица_5_Прочность", index=False)
        table_mlp24.to_excel(writer, sheet_name="Соотношение_MLP24", index=False)
    log.info(f"\n✓ Метрики сохранены в {RESULTS_XLSX}")

    metadata = {
        "version": "3.0",
        "seed": SEED,
        "dataset_shape": list(df.shape),
        "features_23": FEATURES_23,
        "features_24": FEATURES_24,
        "target_modulus": TARGET_MODULUS,
        "target_strength": TARGET_STRENGTH,
        "target_ratio": TARGET_RATIO,
        "best_model_modulus": best_modulus,
        "best_model_strength": best_strength,
        "artifact_filenames": {
            "scaler_23":       "scaler_23.joblib",
            "best_modulus":    "best_modulus.joblib",
            "best_strength":   "best_strength.joblib",
            "mlp_ratio":       "mlp_ratio.pt",
            "scaler_mlp_X":    "scaler_mlp_X.joblib",
            "scaler_mlp_y":    "scaler_mlp_y.joblib",
        },
        "table_4_modulus": table4.to_dict(orient="records"),
        "table_5_strength": table5.to_dict(orient="records"),
        "mlp24": mlp24_info,
        "saved_models_modulus": saved_mod,
        "saved_models_strength": saved_str,
    }
    METADATA_JSON.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    log.info(f"✓ metadata.json сохранён в {METADATA_JSON}")

    log.info("\n" + "=" * 70)
    log.info("ИТОГО (R2 на 10-fold CV / R2 на test):")
    log.info("=" * 70)
    log.info(f"\n{TARGET_MODULUS}:")
    log.info(table4[["Model", "R2_CV", "R2_test", "MAE_test"]].to_string(index=False))
    log.info(f"\n{TARGET_STRENGTH}:")
    log.info(table5[["Model", "R2_CV", "R2_test", "MAE_test"]].to_string(index=False))
    log.info(f"\n{TARGET_RATIO}:")
    log.info(table_mlp24[["Model", "R2_train", "R2_test", "MAE_test"]].to_string(index=False))
    log.info(f"\nВсё выполнено за {time.time() - t_start:.1f} с")


if __name__ == "__main__":
    main()
