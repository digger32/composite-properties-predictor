"""Streamlit-приложение для прогнозирования свойств композиционных материалов.

Приложение загружает обученные модели из директории models/ на основании
описания моделей в файле metadata.json. Это позволяет автоматически
подстраиваться под конкретную модель, выбранную скриптом train_models.py
как лучшую по результатам 10-кратной кросс-валидации.

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
METADATA_PATH = MODELS_DIR / "metadata.json"


# --------------------------------------------------------------------------- #
class MLPRatio(nn.Module):
    """Архитектура должна точно совпадать с train_models.py."""

    def __init__(self, n_in: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
@st.cache_resource
def load_artifacts():
    """Загружает все модели на основе описания в metadata.json."""
    if not METADATA_PATH.exists():
        st.error(
            f"Файл `{METADATA_PATH}` не найден. "
            "Запустите `python train_models.py` для создания обученных моделей."
        )
        st.stop()

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    files = metadata["artifact_filenames"]

    scaler_23 = joblib.load(MODELS_DIR / files["scaler_23"])
    best_modulus = joblib.load(MODELS_DIR / files["best_modulus"])
    best_strength = joblib.load(MODELS_DIR / files["best_strength"])

    mlp = MLPRatio(n_in=len(metadata["features_24"]))
    mlp.load_state_dict(torch.load(MODELS_DIR / files["mlp_ratio"], map_location="cpu"))
    mlp.eval()
    scaler_mlp_X = joblib.load(MODELS_DIR / files["scaler_mlp_X"])
    scaler_mlp_y = joblib.load(MODELS_DIR / files["scaler_mlp_y"])

    return {
        "metadata": metadata,
        "scaler_23": scaler_23,
        "best_modulus": best_modulus,
        "best_strength": best_strength,
        "mlp": mlp,
        "scaler_mlp_X": scaler_mlp_X,
        "scaler_mlp_y": scaler_mlp_y,
    }


# --------------------------------------------------------------------------- #
# Средние значения по датасету — используются как разумные значения по умолчанию
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


def show_model_disclaimer(r2_test: float, context: str) -> None:
    if r2_test < 0.1:
        st.warning(
            f"⚠ **Ограничение модели ({context})**: R²(test) = {r2_test:+.3f}. "
            "Качество прогноза сопоставимо с предсказанием среднего значения "
            "обучающей выборки (baseline). Приложение носит демонстрационный "
            "характер и не предназначено для использования при проектировании "
            "реальных изделий без дополнительной валидации на расширенном "
            "датасете. Подробное обоснование приведено в разделах 2.3 и 2.4 "
            "пояснительной записки."
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

# ------------------------------ Вкладка 1 ---------------------------------- #
with tab1:
    st.subheader("Ввод технологических параметров")
    st.caption(
        f"Модели (выбраны по R²(CV) в подразделе 2.3 пояснительной записки): "
        f"**{metadata['best_model_modulus']}** — для модуля упругости; "
        f"**{metadata['best_model_strength']}** — для прочности."
    )

    cols = st.columns(2)
    inputs_1 = {}
    for i, feat in enumerate(metadata["features_23"]):
        with cols[i % 2]:
            inputs_1[feat] = render_input(feat, "t1")

    left, right = st.columns([1, 1])
    if left.button("↺ Сбросить значения", key="reset_t1"):
        for feat in metadata["features_23"]:
            st.session_state.pop(f"t1::{feat}", None)
        st.rerun()

    if right.button("▶ Выполнить прогноз механических свойств",
                    type="primary", key="predict_t1"):
        X_df = pd.DataFrame([[inputs_1[f] for f in metadata["features_23"]]],
                            columns=metadata["features_23"])
        X_scaled = art["scaler_23"].transform(X_df.values)
        pred_mod = float(art["best_modulus"].predict(X_scaled)[0])
        pred_str = float(art["best_strength"].predict(X_scaled)[0])

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
            st.write("**Таблица 4 — модуль упругости при растяжении, ГПа**")
            st.dataframe(pd.DataFrame(metadata["table_4_modulus"]),
                         use_container_width=True)
            st.write("**Таблица 5 — прочность при растяжении, МПа**")
            st.dataframe(pd.DataFrame(metadata["table_5_strength"]),
                         use_container_width=True)

        # Извлечение R²(test) для предупреждений
        t4 = pd.DataFrame(metadata["table_4_modulus"])
        t5 = pd.DataFrame(metadata["table_5_strength"])
        r2_mod = float(t4.loc[t4["Model"] == metadata["best_model_modulus"],
                              "R2_test"].iloc[0])
        r2_str = float(t5.loc[t5["Model"] == metadata["best_model_strength"],
                              "R2_test"].iloc[0])
        show_model_disclaimer(r2_mod, "модуль упругости")
        show_model_disclaimer(r2_str, "прочность")


# ------------------------------ Вкладка 2 ---------------------------------- #
with tab2:
    st.subheader("Рекомендация соотношения матрица-наполнитель")
    st.caption(
        "Используется нейронная сеть MLP (PyTorch), обученная на 10 "
        "технологических параметрах без утечки данных — модуль упругости и "
        "прочность при растяжении из входов исключены (подраздел 2.4 "
        "пояснительной записки)."
    )

    cols = st.columns(2)
    inputs_2 = {}
    for i, feat in enumerate(metadata["features_24"]):
        with cols[i % 2]:
            inputs_2[feat] = render_input(feat, "t2")

    left, right = st.columns([1, 1])
    if left.button("↺ Сбросить значения", key="reset_t2"):
        for feat in metadata["features_24"]:
            st.session_state.pop(f"t2::{feat}", None)
        st.rerun()

    if right.button("▶ Получить рекомендацию", type="primary", key="predict_t2"):
        X = np.array([[inputs_2[f] for f in metadata["features_24"]]],
                     dtype=np.float32)
        X_scaled = art["scaler_mlp_X"].transform(X).astype(np.float32)
        with torch.no_grad():
            pred_s = art["mlp"](torch.tensor(X_scaled)).numpy().ravel()
        ratio = float(art["scaler_mlp_y"].inverse_transform(
            pred_s.reshape(-1, 1)).ravel()[0])

        st.success("Рекомендация сформирована.")
        st.metric("Рекомендуемое соотношение матрица-наполнитель", f"{ratio:.3f}")

        with st.expander("Метрики модели MLP-2.4 (из metadata.json)"):
            st.json(metadata["mlp24"])

        r2_ratio = float(metadata["mlp24"]["metrics_mlp24"]["R2_test"])
        show_model_disclaimer(r2_ratio, "соотношение матрица-наполнитель")


# ------------------------------ Футер -------------------------------------- #
st.divider()
st.caption(
    f"Версия моделей: {metadata.get('version', '—')} | "
    f"random_state = {metadata.get('seed', '—')} | "
    f"Датасет: {metadata.get('dataset_shape', ['—'])[0]} × "
    f"{metadata.get('dataset_shape', ['—', '—'])[1]}"
)
