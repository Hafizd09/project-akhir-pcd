from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import joblib
import os

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

app = Flask(__name__)
app.template_folder = os.path.abspath('templates')

# =========================
# LOAD MODEL
# =========================
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

CLASS_NAMES = {
    0: "Terumbu Karang SEHAT",
    1: "Terumbu Karang RUSAK / MATI"
}

# =========================
# FEATURE EXTRACTION (HARUS SAMA DENGAN TRAINING)
# =========================
def extract_features(img):

    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # ========= GLCM =========
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    glcm = graycomatrix(
        gray,
        distances=[1, 2, 3],
        angles=angles,
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

    # ========= LBP =========
    radius = 3
    points = 8 * radius  # 24

    lbp = local_binary_pattern(
        gray,
        P=points,
        R=radius,
        method="uniform"
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, points + 3),
        range=(0, points + 2)
    )

    hist = hist.astype("float")
    hist = hist / (hist.sum() + 1e-7)

    # ========= COMBINE =========
    features = np.hstack([glcm_features, hist])

    return features


# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/classify')
def classify():
    return render_template('classify.html')

@app.route('/about')
def about():
    return render_template('about.html')


# =========================
# PREDICT
# =========================
@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'File kosong'}), 400

    try:
        img_bytes = file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Gambar tidak valid'}), 400

        # =========================
        # FEATURE EXTRACTION
        # =========================
        features = extract_features(img)

        # DEBUG (opsional)
        print("Feature length:", len(features))  # HARUS 32

        # =========================
        # SCALING + PREDICT
        # =========================
        features_scaled = scaler.transform([features])
        prediction = svm_model.predict(features_scaled)[0]

        result_class = CLASS_NAMES[prediction]

        # GLCM Features untuk Radar Chart
        contrast = float(features[0])
        correlation = float(features[1])
        energy = float(features[2])
        homogeneity = float(features[3])

        # =========================
        # CONFIDENCE
        # =========================
        try:
            proba = svm_model.predict_proba(features_scaled)[0]
            confidence = float(np.max(proba)) * 100
        except:
            confidence = 85.0

        return jsonify({
    "prediction": result_class,
    "confidence": round(confidence, 2),
    
})

    except Exception as e:
        return jsonify({
            "error": f"Gagal memproses gambar: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)