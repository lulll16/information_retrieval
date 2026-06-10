from flask import Flask, render_template, request, send_file
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import os
import re

app = Flask(__name__)

# ============================================================
# KONFIGURASI DATA
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NAMA_FILE_EXCEL = os.path.join(BASE_DIR, "hasil_scraping_kompas (1).xlsx")

paper = []
konten_list = []
pesan_error_data = ""

try:
    df = pd.read_excel(NAMA_FILE_EXCEL)

    # Format minimal:
    # Kolom 0 = URL
    # Kolom 1 = Judul
    # Kolom 2 = Waktu
    # Kolom 3 = Konten
    if df.shape[1] < 4:
        raise ValueError("File Excel harus memiliki minimal 4 kolom: URL, Judul, Waktu, Konten.")

    paper = df.values.tolist()
    konten_list = [str(doc[3]) for doc in paper]
    print("Data Excel berhasil dibaca.")

except FileNotFoundError:
    pesan_error_data = f"File Excel tidak ditemukan: {NAMA_FILE_EXCEL}"
    print(pesan_error_data)

except Exception as e:
    pesan_error_data = f"Terjadi kesalahan saat membaca Excel: {e}"
    print(pesan_error_data)


# ============================================================
# MODEL TF-IDF
# ============================================================
if konten_list:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(konten_list)
else:
    vectorizer = None
    tfidf_matrix = None


# ============================================================
# FUNGSI BANTUAN
# ============================================================
def ambil_jumlah_berita():
    try:
        jumlah = int(request.form.get("jumlah_berita", 10))
    except ValueError:
        jumlah = 10

    return max(1, min(jumlah, 100))


def ambil_kolom(doc, index, default="-"):
    try:
        nilai = doc[index]
        if pd.isna(nilai):
            return default
        return nilai
    except Exception:
        return default


def nama_file_aman(teks):
    teks = str(teks).strip()
    teks = re.sub(r"[^a-zA-Z0-9_-]+", "_", teks)
    return teks[:50] if teks else "hasil_pencarian"


def normalisasi_teks(teks):
    teks = str(teks).lower()
    teks = re.sub(r"[^a-zA-Z0-9\s]", " ", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    return teks


def token_query(query):
    """
    Mengambil kata penting dari query.
    Kata yang terlalu pendek diabaikan agar evaluasi tidak terlalu longgar.
    """
    query = normalisasi_teks(query)
    tokens = [kata for kata in query.split() if len(kata) >= 3]
    return tokens


# ============================================================
# ACUAN EVALUASI OTOMATIS
# Tidak ada input ground truth manual.
# Sistem otomatis menganggap dokumen relevan jika judul/konten
# mengandung kata dari query.
# ============================================================
def buat_acuan_relevansi_otomatis(query_user):
    tokens = token_query(query_user)

    if not tokens:
        return []

    dokumen_relevan = []

    for index, doc in enumerate(paper):
        judul = str(ambil_kolom(doc, 1, ""))
        konten = str(ambil_kolom(doc, 3, ""))

        gabungan = normalisasi_teks(judul + " " + konten)

        # Relevan otomatis jika minimal salah satu kata query muncul di judul/konten
        if any(kata in gabungan for kata in tokens):
            dokumen_relevan.append(index + 1)

    return dokumen_relevan


# ============================================================
# PENCARIAN DOKUMEN
# ============================================================
def cari_dokumen(query_user, jumlah_berita=10):
    if not query_user or vectorizer is None or tfidf_matrix is None:
        return []

    jumlah_berita = max(1, min(int(jumlah_berita), 100))

    query_vec = vectorizer.transform([query_user])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = similarities.argsort()[::-1][:jumlah_berita]

    hasil = []
    for rank, idx in enumerate(top_indices, start=1):
        skor = float(similarities[idx])

        if skor > 0:
            doc = paper[idx]

            konten = str(ambil_kolom(doc, 3, ""))
            ringkasan = konten[:280] + "..." if len(konten) > 280 else konten

            hasil.append({
                "rank": rank,
                "nomor": idx + 1,
                "url": ambil_kolom(doc, 0, "#"),
                "judul": ambil_kolom(doc, 1, "Tanpa Judul"),
                "waktu": ambil_kolom(doc, 2, "-"),
                "konten": ringkasan,
                "skor": round(skor, 4),
                "skor_persen": round(skor * 100, 2)
            })

    return hasil


# ============================================================
# EVALUASI OTOMATIS
# ============================================================
def hitung_evaluasi_otomatis(hasil_pencarian, acuan_relevansi, total_dokumen):
    acuan_set = set(acuan_relevansi)
    hasil_nomor = [item["nomor"] for item in hasil_pencarian]
    hasil_set = set(hasil_nomor)

    tp = len(hasil_set & acuan_set)
    fp = len(hasil_set - acuan_set)
    fn = len(acuan_set - hasil_set)
    tn = max(total_dokumen - tp - fp - fn, 0)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    # Average Precision
    relevan_ditemukan = 0
    total_precision = 0

    for posisi, nomor in enumerate(hasil_nomor, start=1):
        if nomor in acuan_set:
            relevan_ditemukan += 1
            total_precision += relevan_ditemukan / posisi

    ap = total_precision / len(acuan_set) if acuan_set else 0

    return {
        "relevan": len(acuan_set),
        "tampil": len(hasil_nomor),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "ap": round(ap, 4),
        "precision_pct": round(precision * 100, 1),
        "recall_pct": round(recall * 100, 1),
        "f1_pct": round(f1 * 100, 1),
        "ap_pct": round(ap * 100, 1)
    }


def evaluasi_semua_query_default(jumlah_berita):
    daftar_query = ["Tewas", "Rudal", "Israel"]
    total = len(paper)
    daftar = []

    for query in daftar_query:
        hasil = cari_dokumen(query, jumlah_berita)
        acuan = buat_acuan_relevansi_otomatis(query)
        evaluasi = hitung_evaluasi_otomatis(hasil, acuan, total)
        evaluasi["query"] = query
        daftar.append(evaluasi)

    if daftar:
        rata_precision = sum(item["precision"] for item in daftar) / len(daftar)
        rata_recall = sum(item["recall"] for item in daftar) / len(daftar)
        rata_f1 = sum(item["f1"] for item in daftar) / len(daftar)
        map_score = sum(item["ap"] for item in daftar) / len(daftar)
    else:
        rata_precision = rata_recall = rata_f1 = map_score = 0

    ringkasan = {
        "rata_precision": round(rata_precision, 4),
        "rata_recall": round(rata_recall, 4),
        "rata_f1": round(rata_f1, 4),
        "map": round(map_score, 4),
        "rata_precision_pct": round(rata_precision * 100, 1),
        "rata_recall_pct": round(rata_recall * 100, 1),
        "rata_f1_pct": round(rata_f1 * 100, 1),
        "map_pct": round(map_score * 100, 1)
    }

    return daftar, ringkasan


# ============================================================
# ROUTE UTAMA
# ============================================================
@app.route("/", methods=["GET", "POST"])
def index():
    query_user = ""
    jumlah_berita = 10
    hasil_pencarian = None
    evaluasi = None
    daftar_evaluasi_semua = None
    ringkasan_evaluasi_semua = None
    active_tab = "search"

    if request.method == "POST":
        mode = request.form.get("mode", "cari")
        jumlah_berita = ambil_jumlah_berita()

        if mode == "evaluasi_semua":
            active_tab = "all"
            daftar_evaluasi_semua, ringkasan_evaluasi_semua = evaluasi_semua_query_default(jumlah_berita)

        else:
            active_tab = "search"
            query_user = request.form.get("query", "").strip()
            hasil_pencarian = cari_dokumen(query_user, jumlah_berita)

            acuan_relevansi = buat_acuan_relevansi_otomatis(query_user)
            evaluasi = hitung_evaluasi_otomatis(hasil_pencarian, acuan_relevansi, len(paper))

            acuan_set = set(acuan_relevansi)
            for item in hasil_pencarian:
                item["status_relevansi"] = "Relevan" if item["nomor"] in acuan_set else "Tidak Relevan"

    return render_template(
        "index.html",
        query_user=query_user,
        jumlah_berita=jumlah_berita,
        hasil_pencarian=hasil_pencarian,
        evaluasi=evaluasi,
        daftar_evaluasi_semua=daftar_evaluasi_semua,
        ringkasan_evaluasi_semua=ringkasan_evaluasi_semua,
        total_dokumen=len(paper),
        pesan_error_data=pesan_error_data,
        active_tab=active_tab
    )


# ============================================================
# DOWNLOAD EXCEL
# ============================================================
@app.route("/download", methods=["POST"])
def download():
    query_user = request.form.get("query", "").strip()
    jumlah_berita = ambil_jumlah_berita()

    hasil = cari_dokumen(query_user, jumlah_berita)
    acuan_relevansi = buat_acuan_relevansi_otomatis(query_user)
    acuan_set = set(acuan_relevansi)

    if not hasil:
        return "Tidak ada data yang bisa diunduh."

    for item in hasil:
        item["status_relevansi"] = "Relevan" if item["nomor"] in acuan_set else "Tidak Relevan"

    evaluasi = hitung_evaluasi_otomatis(hasil, acuan_relevansi, len(paper))

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(hasil).to_excel(writer, index=False, sheet_name="Hasil Pencarian")
        pd.DataFrame([evaluasi]).to_excel(writer, index=False, sheet_name="Evaluasi")

    output.seek(0)

    nama = nama_file_aman(query_user)
    return send_file(
        output,
        as_attachment=True,
        download_name=f"hasil_{nama}_{jumlah_berita}_berita.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


if __name__ == "__main__":
    app.run(debug=True)
