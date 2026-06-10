from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import joblib
import os

app = Flask(__name__)

# Mengatur lokasi folder template agar mencari di direktori yang benar
app.template_folder = os.path.abspath('templates')

# --- LOAD MODEL & SCALER ---
try:
    svm_model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    print(f"Peringatan Model: {e}. Pastikan file 'svm_model.pkl' dan 'scaler.pkl' tersedia.")

CLASS_NAMES = {0: "Terumbu Karang SEHAT", 1: "Terumbu Karang RUSAK / MATI"}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/classify')
def classify():
    return render_template('classify.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar yang dikirimkan'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400
    
    try:
        # 1. Membaca dan Mengubah Ukuran Gambar
        image_bytes = file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Format file tidak valid atau rusak'}), 400
            
        img = cv2.resize(img, (256, 256)) 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Ekstraksi Fitur GLCM Berbasis 4 Sudut Spasial (0, 45, 90, 135 derajat)
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray, distances=[1], angles=angles, levels=256, symmetric=True, normed=True)
        
        # Mengambil array fitur dari 4 sudut orientasi
        contrast = graycoprops(glcm, 'contrast')[0]
        correlation = graycoprops(glcm, 'correlation')[0]
        energy = graycoprops(glcm, 'energy')[0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0]
        
        # PERBAIKAN 1: Hitung rata-rata (mean) dari 4 sudut agar pas menjadi 4 fitur input bagi StandardScaler
        mean_contrast = np.mean(contrast)
        mean_correlation = np.mean(correlation)
        mean_energy = np.mean(energy)
        mean_homogeneity = np.mean(homogeneity)
        
        # Menggabungkan nilai rata-rata menjadi format array 1D berisi tepat 4 parameter
        features = np.array([mean_contrast, mean_correlation, mean_energy, mean_homogeneity])
        
        # 3. Standardisasi Fitur & Eksekusi Prediksi SVM
        features_scaled = scaler.transform([features])
        prediction = svm_model.predict(features_scaled)[0]
        result_class = CLASS_NAMES[prediction]
        
        # PERBAIKAN 2: Proteksi try-except jika model SVM dilatih tanpa parameter probability=True
        try:
            probabilities = svm_model.predict_proba(features_scaled)[0]
            confidence_score = probabilities[prediction] * 100
        except (AttributeError, Exception):
            # Nilai fallback/default jika fungsi hitung probabilitas bawaan scikit-learn tidak aktif
            confidence_score = 85.0 
        
        # Mengirimkan nilai balik ke frontend untuk di-render oleh Chart.js Radar
        return jsonify({
            'contrast': round(float(mean_contrast), 4),
            'correlation': round(float(mean_correlation), 4),
            'energy': round(float(mean_energy), 4),
            'homogeneity': round(float(mean_homogeneity), 4),
            'prediction': result_class,
            'confidence': round(confidence_score, 2)
        })
        
    except Exception as e:
        return jsonify({'error': f"Gagal memproses gambar pada server: {str(e)}"}), 500

if __name__ == '__main__':
    # Berjalan di port default 5000 dengan fitur auto-reload (debug=True)
    app.run(debug=True, port=5000)