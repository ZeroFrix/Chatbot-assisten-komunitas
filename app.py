import streamlit as st
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# GANTI DENGAN KODE INI:
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# =====================================================================
# 1. MEMBACA DATASET DARI FILE CSV
# =====================================================================
try:
    df = pd.read_csv("dataset_komunitas.csv")
except FileNotFoundError:
    st.error("File 'dataset_komunitas.csv' tidak ditemukan! Letakkan file tersebut dalam satu folder dengan app.py")
    st.stop()

# =====================================================================
# 2. LOGIKA ML 1: PENCARIAN TANYA JAWAB (Perbaikan Antisipasi Data Kosong)
# =====================================================================
def proses_pencarian_ai(input_user, dataframe):
    # .fillna("") memastikan jika ada kolom pertanyaan yang kosong tidak merusak program
    kumpulan_teks = dataframe['pertanyaan'].fillna("").tolist() + [input_user.lower()]
    tfidf = TfidfVectorizer()
    matriks_vektor = tfidf.fit_transform(kumpulan_teks)
    
    skor_kemiripan = cosine_similarity(matriks_vektor[-1], matriks_vektor[:-1])[0]
    indeks_tercocok = skor_kemiripan.argsort()[-1]
    skor_tertinggi = skor_kemiripan[indeks_tercocok]
    
    if skor_tertinggi > 0.20:
        # Gunakan str() untuk memastikan tipe data yang keluar selalu berupa String (Teks)
        jawaban = str(dataframe['jawaban'].iloc[indeks_tercocok]).strip()
        sumber = str(dataframe['sumber_valid'].iloc[indeks_tercocok]).strip()
        topik = str(dataframe['topik'].iloc[indeks_tercocok]).strip()
        
        # Jika kolom jawaban ternyata kosong/nan di CSV, berikan info default
        if jawaban == "nan" or jawaban == "":
            jawaban = "Maaf, informasi detail untuk pertanyaan ini belum dilengkapi di sistem."
            
        return jawaban, sumber, topik, skor_tertinggi
    return None, None, None, skor_tertinggi

# =====================================================================
# 3. LOGIKA ML 2: TEXT SIMPLIFIER (Extractive Summarization Sederhana)
# =====================================================================
def sederhanakan_teks(teks_panjang, jumlah_kalimat=2):
    # Memecah dokumen panjang menjadi kalimat-kalimat tunggal
    kalimat_list = nltk.sent_tokenize(teks_panjang)
    if len(kalimat_list) <= jumlah_kalimat:
        return teks_panjang
        
    # Menghitung bobot pentingnya kalimat menggunakan TF-IDF
    tfidf = TfidfVectorizer()
    matriks_tfidf = tfidf.fit_transform(kalimat_list)
    
    # Menjumlahkan skor TF-IDF untuk setiap kalimat sebagai indikator tingkat kepentingan
    skor_kalimat = matriks_tfidf.sum(axis=1).A1
    
    # Mengambil indeks kalimat dengan skor tertinggi
    indeks_penting = skor_kalimat.argsort()[-jumlah_kalimat:]
    indeks_penting.sort()  # Mengurutkan kembali sesuai urutan membaca yang asli
    
    kesimpulan = [kalimat_list[i] for i in indeks_penting]
    return " ".join(kesimpulan)

# =====================================================================
# 4. ANTARMUKA APLIKASI (Streamlit UI - Sistem Tab Multi-Fitur)
# =====================================================================
st.set_page_config(page_title="Asisten Komunitas AI V2", page_icon="🤖", layout="wide")

st.title("🤖 Asisten Pintar & Penyederhana Informasi Komunitas")
st.write("Aplikasi Solusi Cerdas untuk Membantu Warga Mengakses Informasi Valid dan Menyederhanakan Regulasi Rumit.")
st.markdown("---")

# Membuat Navigasi Tab Utama aplikasi
tab1, tab2 = st.tabs(["💬 Asisten Tanya Jawab & Anti-Hoaks", "📄 Penyederhana Dokumen Resmi (Simplifier)"])

# ---------------------------------------------------------------------
# TAB 1: ASISTEN CHATBOT & DETEKSI HOAKS
# ---------------------------------------------------------------------
with tab1:
    st.subheader("Tanyakan Prosedur Layanan atau Verifikasi Berita")
    st.write("Masukkan pertanyaan seputar bansos, layanan anak KPAI, posko PMI, atau pesan berantai yang Anda terima.")
    
    teks_masukan = st.text_input("Masukkan pertanyaan Anda:", placeholder="Contoh: apakah pendaftaran bansos lewat whatsapp itu asli?")
    
    if teks_masukan:
        jawaban, sumber, topik, akurasi = proses_pencarian_ai(teks_masukan, df)
        
        if jawaban:
            st.markdown(f"#### 📋 Hasil Analisis AI (Kategori: {topik})")
            
            # Deteksi bahaya hoaks secara visual
            if "HOAKS ALERT!" in jawaban:
                st.error(jawaban)
            else:
                st.success(jawaban)
                
            # Transparansi Informasi (Responsible AI)
            st.info(f"🌐 **Sumber Valid Terverifikasi:** [{sumber}](https://{sumber})")
            st.caption(f"Tingkat kecocokan model: {akurasi*100:.2f}%")
        else:
            st.warning("⚠️ Informasi tidak ditemukan di sistem lokal. Tetap waspada terhadap informasi yang belum jelas keabsahannya.")

# ---------------------------------------------------------------------
# TAB 2: TEXT SIMPLIFIER (Merangkum Aturan Birokrasi yang Panjang)
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Penyederhana Teks Regulasi & Prosedur Panjang")
    st.write("Warga sering malas membaca aturan atau pengumuman yang panjang lebar. Tempelkan teks panjang tersebut di sini untuk dirangkum otomatis oleh AI menjadi langkah inti.")
    
    # Contoh teks panjang birokrasi sebagai placeholder
    contoh_teks = (
        "Berdasarkan Surat Keputusan Bersama mengenai standarisasi penyaluran jaminan sosial nasional, "
        "setiap warga masyarakat diwajibkan untuk terlebih dahulu melakukan registrasi pada sistem DTKS kementerian. "
        "Prosedur ini melibatkan validasi berlapis dimulai dari tingkat RT hingga kelurahan setempat. "
        "Setelah data terverifikasi di kelurahan, barulah berkas fisik berupa KTP dan KK asli dikirimkan ke Dinas Sosial. "
        "Masyarakat dilarang keras memberikan imbalan uang kepada petugas lapangan selama proses penyaluran berlangsung."
    )
    
    teks_panjang_input = st.text_area(
        "Tempel berkas/pengumuman resmi yang panjang di sini:", 
        value=contoh_teks, 
        height=150
    )
    
    jumlah_output = st.slider("Pilih jumlah kalimat ringkasan inti yang diinginkan:", min_value=1, max_value=3, value=2)
    
    if st.button("Sederhanakan Teks Sekarang"):
        with st.spinner("AI sedang merangkum poin penting..."):
            hasil_ringkas = sederhanakan_teks(teks_panjang_input, jumlah_kalimat=jumlah_output)
            
            st.markdown("#### 🎯 Hasil Poin Inti (Aman & Mudah Dipahami):")
            st.success(hasil_ringkas)
            st.caption("ℹ️ *Fitur ini menggunakan algoritma Extractive Summarization berbasis TF-IDF untuk menyaring kalimat yang paling mengandung informasi kunci.*")

# Catatan Kaki Aplikasi
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Ekshibisi KA/AI - LKS Dikmen Tingkat Nasional 2026</p>", unsafe_allow_html=True)