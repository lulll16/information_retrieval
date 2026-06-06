from flask import Flask, render_template, request, send_file
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import os
import re

app = Flask(__name__)

# ============================================================
# 1. MEMBACA DATA EXCEL
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
nama_file_excel = os.path.join(BASE_DIR, "hasil_scraping_kompas (1).xlsx")

paper = []
konten_list = []

try:
    df = pd.read_excel(nama_file_excel)

    # Pastikan file Excel minimal punya 4 kolom:
    # kolom 0 = URL, kolom 1 = Judul, kolom 2 = Waktu, kolom 3 = Konten
    if df.shape[1] < 4:
        raise ValueError("File Excel harus memiliki minimal 4 kolom: URL, Judul, Waktu, Konten.")

    paper = df.values.tolist()
    konten_list = [str(doc[3]) for doc in paper]

    print("Data Excel berhasil dibaca.")

except FileNotFoundError:
    print(f"File Excel tidak ditemukan: {nama_file_excel}")

except Exception as e:
    print(f"Terjadi kesalahan saat membaca Excel: {e}")


# ============================================================
# 2. MEMBUAT MODEL TF-IDF
# ============================================================
if len(konten_list) > 0:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(konten_list)
else:
    vectorizer = None
    tfidf_matrix = None


# ============================================================
# 3. FUNGSI BANTUAN
# ============================================================
def get_jumlah_berita():
    """
    Mengambil jumlah berita dari form.
    Nilai dibatasi dari 1 sampai 100.
    """
    try:
        jumlah = int(request.form.get("jumlah_berita", 5))
    except ValueError:
        jumlah = 5

    if jumlah < 1:
        jumlah = 1

    if jumlah > 100:
        jumlah = 100

    return jumlah


def ambil_kolom(doc, index, default="-"):
    """
    Mengambil data berdasarkan index kolom.
    Fungsi ini dibuat agar aplikasi tidak error jika ada kolom kosong.
    """
    try:
        nilai = doc[index]
        if pd.isna(nilai):
            return default
        return nilai
    except Exception:
        return default


def nama_file_aman(teks):
    """
    Membuat nama file download agar aman.
    """
    teks = str(teks).strip()
    teks = re.sub(r"[^a-zA-Z0-9_-]+", "_", teks)
    return teks[:50] if teks else "hasil_pencarian"


# ============================================================
# 4. FUNGSI PENCARIAN DOKUMEN
# ============================================================
def cari_dokumen(query_user, jumlah_berita=5):
    if query_user == "" or vectorizer is None or tfidf_matrix is None:
        return []

    jumlah_berita = max(1, min(int(jumlah_berita), 100))

    query_vec = vectorizer.transform([query_user])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Mengurutkan dokumen dari skor paling tinggi
    top_indices = similarities.argsort()[::-1][:jumlah_berita]

    hasil = []

    for i in top_indices:
        skor = similarities[i]

        # Hanya tampilkan berita yang punya kemiripan
        if skor > 0:
            doc = paper[i]

            url = ambil_kolom(doc, 0, "#")
            judul = ambil_kolom(doc, 1, "Tanpa Judul")
            waktu = ambil_kolom(doc, 2, "-")
            konten = str(ambil_kolom(doc, 3, ""))

            if len(konten) > 250:
                konten = konten[:250] + "..."

            hasil.append({
                "url": url,
                "judul": judul,
                "waktu": waktu,
                "konten": konten,
                "skor": round(float(skor), 4)
            })

    return hasil


# ============================================================
# 5. HALAMAN UTAMA
# ============================================================
@app.route("/", methods=["GET", "POST"])
def index():
    hasil_pencarian = None
    query_user = ""
    jumlah_berita = 5

    if request.method == "POST":
        query_user = request.form.get("query", "").strip()
        jumlah_berita = get_jumlah_berita()
        hasil_pencarian = cari_dokumen(query_user, jumlah_berita)

    return render_template(
        "index.html",
        hasil_pencarian=hasil_pencarian,
        query_user=query_user,
        jumlah_berita=jumlah_berita
    )


# ============================================================
# 6. DOWNLOAD HASIL KE EXCEL
# ============================================================
@app.route("/download", methods=["POST"])
def download():
    query_user = request.form.get("query", "").strip()
    jumlah_berita = get_jumlah_berita()

    hasil_pencarian = cari_dokumen(query_user, jumlah_berita)

    if not hasil_pencarian:
        return "Tidak ada data yang bisa diunduh."

    df_hasil = pd.DataFrame(hasil_pencarian)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_hasil.to_excel(writer, index=False, sheet_name="Hasil Pencarian")

    output.seek(0)

    nama_query = nama_file_aman(query_user)
    nama_download = f"hasil_{nama_query}_{jumlah_berita}_berita.xlsx"

    return send_file(
        output,
        as_attachment=True,
        download_name=nama_download,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ============================================================
# 7. MENJALANKAN APLIKASI LOKAL
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)
