
import pandas as pd
import numpy as np

PATH_RAW_DATA = 'namadataset_raw/train.csv'
PATH_PROCESSED_DATA = 'namadataset_preprocessing/processed_train.csv'

def muat_data(path):
    print(f"Memuat data mentah dari {path}...")
    return pd.read_csv(path)

def preprocessing_data(df):
    print("Memulai preprocessing data...")
    
    median_age = df['Age'].median()
    df['Age'].fillna(median_age, inplace=True)
    
    mode_embarked = df['Embarked'].mode()[0]
    df['Embarked'].fillna(mode_embarked, inplace=True)
    
    df.drop('Cabin', axis=1, inplace=True)
    
    df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
    df = pd.get_dummies(df, columns=['Embarked'])
    
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    
    print("Preprocessing selesai.")
    return df

def simpan_data(df, path):
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
