import pandas as pd
import numpy as np

PATH_RAW_DATA = 'namadataset_raw/train.csv'
PATH_PROCESSED_DATA = 'namadataset_preprocessing/processed_train.csv'

def muat_data(path):
    """Memuat data dari file CSV."""
    print(f"Memuat data mentah dari {path}...")
    return pd.read_csv(path)

def preprocessing_data(df):
    """Membersihkan dan memproses data (logika dari Langkah 5)."""
    print("Memulai preprocessing data...")

    # 1. Menangani Data Kosong
    median_age = df['Age'].median()
    df['Age'].fillna(median_age, inplace=True)
    print(f"Mengisi 'Age' yang kosong dengan median: {median_age}")

    mode_embarked = df['Embarked'].mode()[0]
    df['Embarked'].fillna(mode_embarked, inplace=True)
    print(f"Mengisi 'Embarked' yang kosong dengan modus: {mode_embarked}")

    # 2. Menghapus Kolom (termasuk 'Cabin')
    # 'Cabin' dihapus karena terlalu banyak data kosong
    # 'PassengerId', 'Name', 'Ticket' dihapus karena tidak relevan
    # Kolom 'Survived' TIDAK dihapus.
    df.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    print("Menghapus kolom 'Cabin', 'PassengerId', 'Name', dan 'Ticket'.")

    # 3. Encoding Data Kategorikal
    df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
    df = pd.get_dummies(df, columns=['Embarked'])

    print("Preprocessing selesai.")
    return df

def simpan_data(df, path):
    """Fungsi untuk menyimpan data bersih ke file CSV."""
    print(f"Menyimpan data bersih ke {path}...")
    df.to_csv(path, index=False)
    print("Data bersih berhasil disimpan.")

if __name__ == '__main__':
    try:
        data_mentah = muat_data(PATH_RAW_DATA)
        data_bersih = preprocessing_data(data_mentah)
        simpan_data(data_bersih, PATH_PROCESSED_DATA)
        print("\n--- Alur Kerja Otomatis Selesai ---")
    except FileNotFoundError:
        print(f"Error: File {PATH_RAW_DATA} tidak ditemukan.")
    except Exception as e:
        print(f"Terjadi error tak terduga: {e}")
