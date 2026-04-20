"""Streamlit-приложение для прогнозирования свойств композиционных материалов.

Приложение загружает предварительно обученные модели из директории models/
и предоставляет два независимых режима работы:
    1) прогноз модуля упругости при растяжении и прочности при растяжении;
    2) рекомендацию соотношения матрица-наполнитель.

Запуск:
    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

MODELS_DIR = Path("models")


class MLPRatio(nn.Module):
    """Архитектура MLP-2.4 должна соответствовать train_models.py."""

    def __init__(self, n_in: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@st.cache_resource
def load_artifacts():
    """Загружает все модели один раз и кеширует до перезапуска процесса."""
    if not MODELS_DIR.exists():
        st.error(
            "Директория `models/` не найдена. "
            "Запустите `python train_models.py` для создания моделей."
        )
        st.stop()

    metadata = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    ridge_mod = joblib.load(MODELS_DIR / "ridge_modulus.joblib")
    rf_str = joblib.load(MODELS_DIR / "rf_strength.joblib")
    scaler_23 = joblib.load(MODELS_DIR / "scaler_23.joblib")

    mlp = MLPRatio(n_in=len(metadata["features_24"]))
    mlp.load_state_dict(torch.load(MODELS_DIR / "mlp_ratio.pt", map_location="cpu"))
    mlp.eval()
    scaler_mlp_X = joblib.load(MODELS_DIR / "scaler_mlp_X.joblib")
    scaler_mlp_y = joblib.load(MODELS_DIR / "scaler_mlp_y.joblib")

    return {
        "metadata": metadata,
        "ridge_mod": ridge_mod,
        "rf_str": rf_str,
        "scaler_23": scaler_23,
        "mlp": mlp,
        "scaler_mlp_X": scaler_mlp_X,
        "scaler_mlp_y": scaler_mlp_y,
    }


# Реалистичные значения по умолчанию — средние по обучающей выборке
DEFAULTS = {
    "Соотношение матрица-наполнитель": 2.93,
    "Плотность, кг/м3": 1975.0,
    "модуль упругости, ГПа": 740.0,
    "Количество отвердителя, м.%": 110.6,
    "Содержание эпоксидных групп,%_2": 22.2,
    "Температура вспышки, С_2": 285.9,
    "Поверхностная плотность, г/м2": 482.7,
    "Потребление смолы, г/м2": 218.4,
    "Угол нашивки, град": 0,
    "Шаг нашивки": 6.9,
    "Плотность нашивки": 57.2,
}

INPUT_SPECS = {
    "Соотношение матрица-наполнитель": dict(min=0.0, max=10.0, step=0.1, fmt="%.2f"),
    "Плотность, кг/м3": dict(min=1500.0, max=2500.0, step=5.0, fmt="%.1f"),
    "модуль упругости, ГПа": dict(min=0.0, max=2000.0, step=5.0, fmt="%.1f"),
    "Количество отвердителя, м.%": dict(min=0.0, max=250.0, step=1.0, fmt="%.1f"),
    "Содержание эпоксидных групп,%_2": dict(min=10.0, max=40.0, step=0.1, fmt="%.2f"),
    "Температура вспышки, С_2": dict(min=50.0, max=500.0, step=1.0, fmt="%.1f"),
    "Поверхностная плотность, г/м2": dict(min=0.0, max=1500.0, step=5.0, fmt="%.1f"),
    "Потребление смолы, г/м2": dict(min=0.0, max=500.0, step=1.0, fmt="%.1f"),
    "Угол нашивки, град": dict(options=[0, 90]),
    "Шаг нашивки": dict(min=0.0, max=20.0, step=0.1, fmt="%.1f"),
    "Плотность нашивки": dict(min=0.0, max=150.0, step=1.0, fmt="%.1f"),
}


def render_input(feature: str, key_prefix: str):
    """Отображает поле ввода, соответствующее типу признака."""
    spec = INPUT_SPECS[feature]
    key = f"{key_prefix}::{feature}"
    if "options" in spec:
        return st.selectbox(feature, options=spec["options"], key=key)
    return st.number_input(
        feature,
        min_value=spec["min"], max_value=spec["max"],
        value=float(DEFAULTS[feature]), step=spec["step"], format=spec["fmt"],
        key=key,
    )


def show_model_disclaimer(metrics: dict, context: str) -> None:
    """Отображает честное предупреждение о качестве модели."""
    r2 = metrics.get("R2_test", metrics.get("R2_CV", 0))
    if r2 < 0.1:
        st.warning(
            f"⚠ **Ограничение модели ({context})**: R²(test) = {r2:+.3f}. "
            "Качество прогноза сопоставимо с предсказанием среднего значения "
            "обучающей выборки (baseline). Приложение носит демонстрационный "
            "характер и **не предназначено** для использования при проектировании "
            "реальных изделий без дополнительной валидации на расширенном датасете. "
            "Подробное обоснование приведено в разделах 2.3 и 2.4 пояснительной "
            "записки."
        )


# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Прогноз свойств композитов", layout="wide")
st.title("Прогнозирование конечных свойств композиционных материалов")
st.caption(
    "Выпускная квалификационная работа по курсу «Data Science Pro», "
    "МГТУ им. Н. Э. Баумана, 2025"
)

art = load_artifacts()
metadata = art["metadata"]

tab1, tab2 = st.tabs(
    ["📊 Прогноз механических свойств",
     "🎛 Рекомендация соотношения матрица-наполнитель"]
)

# ---------------------------- Вкладка 1 ------------------------------------ #
with tab1:
    st.subheader("Ввод технологических параметров")
    st.caption(
        "Используются модели Ridge (для модуля упругости) и Random Forest "
        "(для прочности), выбранные по результатам 10-кратной кросс-валидации "
        "(подраздел 2.3 пояснительной записки)."
    )

    cols = st.columns(2)
    inputs_1 = {}
    for i, feat in enumerate(metadata["features_23"]):
        with cols[i % 2]:
            inputs_1[feat] = render_input(feat, "t1")

    left, right = st.columns([1, 1])
    reset = left.button("↺ Сбросить значения", key="reset_t1")
    if reset:
        for feat in metadata["features_23"]:
            st.session_state.pop(f"t1::{feat}", None)
        st.rerun()

    predict = right.button(
        "▶ Выполнить прогноз механических свойств", type="primary", key="predict_t1"
    )
    if predict:
        X_df = pd.DataFrame([[inputs_1[f] for f in metadata["features_23"]]],
                            columns=metadata["features_23"])
        X_scaled = art["scaler_23"].transform(X_df.values)
        pred_mod = float(art["ridge_mod"].predict(X_scaled)[0])
        pred_str = float(art["rf_str"].predict(X_scaled)[0])

        st.success("Прогноз выполнен.")
        m1, m2 = st.columns(2)
        m1.metric("Модуль упругости при растяжении, ГПа", f"{pred_mod:.2f}")
        m2.metric("Прочность при растяжении, МПа", f"{pred_str:.1f}")

        result_df = pd.DataFrame({
            "Параметр": ["Модуль упругости при растяжении, ГПа",
                         "Прочность при растяжении, МПа"],
            "Прогноз": [round(pred_mod, 2), round(pred_str, 1)],
        })
        st.download_button(
            "💾 Скачать результат (CSV)",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_mechanical.csv",
            mime="text/csv",
        )

        with st.expander("Метрики моделей (из metadata.json)"):
            st.json(metadata["metrics"])

        show_model_disclaimer(metadata["metrics"]["modulus"],
                              "модуль упругости")
        show_model_disclaimer(metadata["metrics"]["strength"], "прочность")


# ---------------------------- Вкладка 2 ------------------------------------ #
with tab2:
    st.subheader("Рекомендация соотношения матрица-наполнитель")
    st.caption(
        "Используется нейронная сеть MLP (PyTorch), обученная на 10 технологических "
        "параметрах без утечки данных — модуль упругости и прочность при растяжении "
        "из входов исключены (подраздел 2.4 пояснительной записки)."
    )

    cols = st.columns(2)
    inputs_2 = {}
    for i, feat in enumerate(metadata["features_24"]):
        with cols[i % 2]:
            inputs_2[feat] = render_input(feat, "t2")

    left, right = st.columns([1, 1])
    reset = left.button("↺ Сбросить значения", key="reset_t2")
    if reset:
        for feat in metadata["features_24"]:
            st.session_state.pop(f"t2::{feat}", None)
        st.rerun()

    predict = right.button(
        "▶ Получить рекомендацию", type="primary", key="predict_t2"
    )
    if predict:
        X = np.array([[inputs_2[f] for f in metadata["features_24"]]],
                     dtype=np.float32)
        X_scaled = art["scaler_mlp_X"].transform(X).astype(np.float32)
        with torch.no_grad():
            pred_s = art["mlp"](torch.tensor(X_scaled)).numpy().ravel()
        ratio = float(art["scaler_mlp_y"].inverse_transform(
            pred_s.reshape(-1, 1)).ravel()[0])

        st.success("Рекомендация сформирована.")
        st.metric("Рекомендуемое соотношение матрица-наполнитель", f"{ratio:.3f}")

        with st.expander("Метрики модели MLP (из metadata.json)"):
            st.json(metadata["metrics"]["ratio"])

        show_model_disclaimer(metadata["metrics"]["ratio"],
                              "соотношение матрица-наполнитель")


# ---------------------------- Футер --------------------------------------- #
st.divider()
st.caption(
    f"Версия обученных моделей: {metadata.get('version', '—')} | "
    f"random_state = {metadata.get('seed', '—')}"
)
