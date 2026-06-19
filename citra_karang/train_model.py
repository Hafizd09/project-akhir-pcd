import os
import cv2
import joblib
import numpy as np

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# KONFIGURASI
CATEGORIES = ["sehat", "rusak"]

IMAGE_SIZE = (256, 256)

LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS

# FEATURE EXTRACTION
def extract_features(image):

    # Resize
    image = cv2.resize(image, IMAGE_SIZE)

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Histogram Equalization
    gray = cv2.equalizeHist(gray)

   
    # GLCM FEATURES
   

    glcm = graycomatrix(
        gray,
        distances=[1, 2, 3],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    asm = graycoprops(glcm, 'ASM').mean()

    glcm_features = [
        contrast,
        correlation,
        energy,
        homogeneity,
        dissimilarity,
        asm
    ]

    # LBP FEATURES
    lbp = local_binary_pattern(
        gray,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method="uniform"
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_POINTS + 3),
        range=(0, LBP_POINTS + 2)
    )

    hist = hist.astype("float")

    hist /= (hist.sum() + 1e-7)

    # Gabungkan fitur
    features = np.hstack([glcm_features, hist])

    return features

# LOAD DATASET
def load_dataset(base_dir):

    X = []
    y = []

    for label, category in enumerate(CATEGORIES):

        folder = os.path.join(base_dir, category)

        if not os.path.exists(folder):
            print(f"Folder tidak ditemukan: {folder}")
            continue

        files = os.listdir(folder)

        print(f"{category} : {len(files)} gambar")

        for file in files:

            path = os.path.join(folder, file)

            img = cv2.imread(path)

            if img is None:
                continue

            try:
                features = extract_features(img)

                X.append(features)
                y.append(label)

            except Exception as e:
                print("Error:", path)
                print(e)

    return np.array(X), np.array(y)

# MAIN
print("=" * 50)
print("MEMBACA DATA TRAIN")
print("=" * 50)

X_train, y_train = load_dataset("dataset/train")

print("\nMEMBACA DATA TEST")

X_test, y_test = load_dataset("dataset/test")

print("\nJumlah Data Train :", len(X_train))
print("Jumlah Data Test  :", len(X_test))


# NORMALISASI
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# GRID SEARCH SVM
print("\nMENCARI PARAMETER TERBAIK SVM...")

param_grid = {
    'C': [1, 10, 100],
    'gamma': [0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(
    SVC(probability=True),
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

print("\nBest Parameter:")
print(grid.best_params_)

model = grid.best_estimator_

# EVALUASI
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("HASIL EVALUASI")
print("=" * 50)

print(f"Akurasi : {acc * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# SIMPAN MODEL
joblib.dump(model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel berhasil disimpan!")
print("svm_model.pkl")
print("scaler.pkl")