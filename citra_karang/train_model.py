import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

CATEGORIES = ['sehat', 'rusak'] # 0 = sehat, 1 = rusak

# Fungsi untuk mengekstrak fitur GLCM dari sebuah folder
def load_data_from_folder(base_dir):
    X_data = []
    y_label = []
    
    for category_idx, category_name in enumerate(CATEGORIES):
        folder_path = os.path.join(base_dir, category_name)
        if not os.path.exists(folder_path):
            print(f"[Peringatan] Folder tidak ditemukan: {folder_path}")
            continue
            
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue # Lewati jika file rusak/bukan gambar
                
            # Proses Grayscale & Ekstraksi GLCM
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            
            X_data.append([contrast, correlation, energy, homogeneity])
            y_label.append(category_idx)
            
    return np.array(X_data), np.array(y_label)

# 1. Ambil Data dari Folder Masing-Masing
print("Membaca dan mengekstrak DATA TRAINING...")
X_train, y_train = load_data_from_folder('dataset/train')

print("Membaca dan mengekstrak DATA TESTING...")
X_test, y_test = load_data_from_folder('dataset/test')

# Validasi jika data kosong
if len(X_train) == 0 or len(X_test) == 0:
    print("\n[ERROR] Data training atau testing kosong! Periksa kembali folder dataset kamu.")
else:
    # 2. Normalisasi Ciri menggunakan StandardScaler
    scaler = StandardScaler()
    # Fit & Transform berdasarkan data training saja (bocoran data testing harus dihindari)
    X_train_scaled = scaler.fit_transform(X_train) 
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Latih Model SVM dengan Data Latih
    print("\nMelatih model SVM...")
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    # 4. Uji Model dengan Data Uji Khusus Kamu
    y_pred = svm_model.predict(X_test_scaled)
    akurasi = accuracy_score(y_test, y_pred)
    
    # 5. Tampilkan Hasil Akhir
    print("\n" + "="*40)
    print(f" HASIL EVALUASI DATA MANUAK KAMU")
    print("="*40)
    print(f"Total Data Training : {len(X_train)} gambar")
    print(f"Total Data Testing  : {len(X_test)} gambar")
    print(f"AKURASI MODEL SVM   : {akurasi * 100:.2f}%")
    print("="*40 + "\n")
    
    # 6. Simpan Model dan Scaler
    joblib.dump(svm_model, 'svm_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model 'svm_model.pkl' & 'scaler.pkl' siap digunakan oleh Flask!")