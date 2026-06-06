from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

app = Flask(__name__)

# 1. BACA DATA EXCEL
# Pastikan nama file ini sudah sesuai dengan yang ada di foldermu
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
nama_file_excel = os.path.join(BASE_DIR, 'hasil_scraping_kompas (1).xlsx')

try:
    df = pd.read_excel(nama_file_excel)
    paper = df.values.tolist()
    # Mengambil kolom konten saja (indeks ke-3)
    konten_list = [str(doc[3]) for doc in paper]
    print(f"✅ BERHASIL: File '{nama_file_excel}' siap digunakan.")
except FileNotFoundError:
    print(f"❌ ERROR: File '{nama_file_excel}' tidak ditemukan di folder ini!")
    paper = []
    konten_list = []

# 2. INISIALISASI TF-IDF
# Hanya dijalankan jika data konten tidak kosong
if konten_list:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(konten_list)
else:
    vectorizer = None
    tfidf_matrix = None

# 3. FUNGSI MESIN PENCARIAN
def cari_dokumen(query_user):
    if not query_user or not vectorizer:
        return []

    query_vec = vectorizer.transform([query_user])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Ambil 5 dokumen paling mirip
    top_indices = similarities.argsort()[:-6:-1]
    
    hasil = []
    for i in top_indices:
        skor = similarities[i]
        if skor > 0:
            doc = paper[i]
            hasil.append({
                'url': doc[0],
                'judul': doc[1],
                'waktu': doc[2],
                'konten': str(doc[3])[:200] + "...",
                'skor': round(skor, 4)
            })
    return hasil

# 4. HALAMAN UTAMA (WEB)
@app.route('/', methods=['GET', 'POST'])
def index():
    hasil_pencarian = None
    query_masuk = ""
    
    if request.method == 'POST':
        query_masuk = request.form['query']
        hasil_pencarian = cari_dokumen(query_masuk)
        
    return render_template('index.html', hasil_pencarian=hasil_pencarian, query_user=query_masuk)

# 5. FITUR DOWNLOAD EXCEL
@app.route('/download', methods=['POST'])
def download():
    query_masuk = request.form['query']
    
    # Memanggil fungsi pencarian
    hasil_pencarian = cari_dokumen(query_masuk)
    
    if not hasil_pencarian:
        return "Tidak ada data untuk diunduh."

    # Proses pembuatan file Excel di memori
    df_hasil = pd.DataFrame(hasil_pencarian)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_hasil.to_excel(writer, index=False, sheet_name='Hasil Pencarian')
    output.seek(0)
    
    nama_file = f"hasil_{query_masuk}.xlsx"
    
    return send_file(
        output,
        as_attachment=True,
        download_name=nama_file,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# 6. JALANKAN SERVER
if __name__ == '__main__':
    app.run(debug=True)