import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
ZERO_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
FEATURE_COLS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']


def load_data(filepath: str) -> pd.DataFrame:
    logger.info(f"Memuat dataset dari: {filepath}")
    df = pd.read_csv(filepath, names=COLUMNS)
    logger.info(f"Dataset berhasil dimuat. Shape: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Menangani missing values...")
    df = df.copy()

    for col in ZERO_COLS:
        df[col] = df[col].replace(0, np.nan)

    for col in ZERO_COLS:
        median_0 = df.loc[df['Outcome'] == 0, col].median()
        median_1 = df.loc[df['Outcome'] == 1, col].median()
        df.loc[(df['Outcome'] == 0) & (df[col].isnull()), col] = median_0
        df.loc[(df['Outcome'] == 1) & (df[col].isnull()), col] = median_1

    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values setelah imputasi: {missing_after}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    logger.info(f"Duplikat dihapus: {before - after} baris. Shape: {df.shape}")
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Menangani outlier dengan metode IQR (capping)...")
    df = df.copy()
    for col in FEATURE_COLS:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)
    logger.info("Outlier berhasil ditangani.")
    return df


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Melakukan standarisasi fitur...")
    df = df.copy()
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    logger.info("Standarisasi selesai.")
    return df


def save_data(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset preprocessing disimpan ke: {output_path}")


def run_preprocessing(input_path: str, output_path: str) -> pd.DataFrame:
    logger.info("=" * 50)
    logger.info("Memulai pipeline preprocessing otomatis")
    logger.info("=" * 50)

    df = load_data(input_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df = normalize_features(df)
    save_data(df, output_path)

    logger.info("=" * 50)
    logger.info(f"Preprocessing selesai! Shape akhir: {df.shape}")
    logger.info("=" * 50)
    return df


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_PATH = os.path.join(BASE_DIR, "diabetes_raw.csv")
    OUTPUT_PATH = os.path.join(BASE_DIR, "preprocessing", "diabetes_preprocessing.csv")

    df_result = run_preprocessing(INPUT_PATH, OUTPUT_PATH)
    print(f"\nPreview hasil preprocessing:")
    print(df_result.head())
