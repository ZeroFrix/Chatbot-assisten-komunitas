import streamlit as st
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================================
# 1. INISIALISASI RESOURCE & DOWNLOAD DATA (CACHE)
# =====================================================================
@st.cache_resource(show_spinner=False)
def inisialisasi_nltk():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

inisialisasi_nltk()

@st.cache_data(show_spinner=False)
def muat_dataset(nama_file="dataset_komunitas.csv"):
    try:
        dataframe = pd.read_csv(nama_file)
        
        # 1. Pengecekan Wajib untuk Kolom 'pertanyaan'
        if 'pertanyaan' in dataframe.columns:
            dataframe['pertanyaan'] = dataframe['pertanyaan'].fillna("").astype(str)
        else:
            st.error("❌ Kolom 'pertanyaan' tidak ditemukan di dataset CSV Anda!")
            st.stop()
            
        # 2. Pengecekan Wajib untuk Kolom 'jawaban'
        if 'jawaban' in dataframe.columns:
            dataframe['jawaban'] = dataframe['jawaban'].fillna("").astype(str)
        else:
            st.error("❌ Kolom 'jawaban' tidak ditemukan di dataset CSV Anda!")
            st.stop()
            
        # 3. Pengecekan Opsional untuk Kolom 'topik' (Jika tidak ada, buat otomatis)
        if 'topik' in dataframe.columns:
            dataframe['topik'] = dataframe['topik'].fillna("Umum").astype(str)
        else:
            dataframe['topik'] = "Umum"
            
        # 4. Pengecekan Opsional untuk Kolom 'sumber_valid' (Jika tidak ada, buat otomatis)
        if 'sumber_valid' in dataframe.columns:
            dataframe['sumber_valid'] = dataframe['sumber_valid'].fillna("Kanal Resmi Instansi").astype(str)
        else:
            dataframe['sumber_valid'] = "Kanal Resmi Instansi"
            
        return dataframe
    except FileNotFoundError:
        return None
df = muat_dataset()

if df is None:
    st.error("❌ File 'dataset_komunitas.csv' tidak ditemukan! Pastikan file berada di folder yang sama dengan app.py")
    st.stop()

# =====================================================================
# 2. MODEL ENGINE 1: FAQ MATCHING (OPTIMIZED)
# =====================================================================
@st.cache_resource(show_spinner=False)
def latih_vektor_faq(daftar_pertanyaan):
    # Melatih TfidfVectorizer hanya sekali untuk basis data
    vectorizer = TfidfVectorizer(lowercase=True)
    matriks_basis = vectorizer.fit_transform(daftar_pertanyaan)
    return vectorizer, matriks_basis

# Buat model pencarian berbasis memori cache
vectorizer_faq, matriks_faq = latih_vektor_faq(df['pertanyaan'].tolist())

def proses_pencarian_ai(input_user, dataframe, vectorizer, matriks_basis):
    if not input_user.strip():
        return None, None, None, 0.0
        
    # Transformasi input user menggunakan kosakata yang sudah dilatih (efisien)
    vektor_user = vectorizer.transform([input_user.lower()])
    
    # Hitung kemiripan secara langsung
    skor_kemiripan = cosine_similarity(vektor_user, matriks_basis)[0]
    indeks_tercocok = skor_kemiripan.argsort()[-1]
    skor_tertinggi = skor_kemiripan[indeks_tercocok]
    
    # Batas ambang keyakinan model (Threshold)
    if skor_tertinggi > 0.20:
        jawaban = dataframe['jawaban'].iloc[indeks_tercocok].strip()
        sumber = dataframe['sumber_valid'].iloc[indeks_tercocok].strip()
        topik = dataframe['topik'].iloc[indeks_tercocok].strip()
        
        if jawaban == "nan" or not jawaban:
            jawaban = "Maaf, informasi detail mengenai topik ini sedang dalam proses verifikasi sistem."
            
        return jawaban, sumber, topik, skor_tertinggi
    return None, None, None, skor_tertinggi

# =====================================================================
# 3. MODEL ENGINE 2: TEXT SIMPLIFIER (EFFICIENT)
# =====================================================================
def sederhanakan_teks(teks_panjang, jumlah_kalimat=2):
    if not teks_panjang.strip():
        return "Silakan masukkan dokumen terlebih dahulu."
        
    kalimat_list = nltk.sent_tokenize(teks_panjang)
    if len(kalimat_list) <= jumlah_kalimat:
        return teks_panjang
        
    # Ekstraksi kalimat penting menggunakan pembobotan TF-IDF tingkat kalimat
    tfidf = TfidfVectorizer()
    matriks_tfidf = tfidf.fit_transform(kalimat_list)
    skor_kalimat = matriks_tfidf.sum(axis=1).A1
    
    indeks_penting = skor_kalimat.argsort()[-jumlah_kalimat:]
    indeks_penting.sort()  # Urutkan sesuai alur bacaan dokumen asli
    
    kesimpulan = [kalimat_list[i] for i in indeks_penting]
    return " ".join(kesimpulan)

# =====================================================================
# 4. ANTARMUKA APLIKASI (STREAMLIT UI)
# =====================================================================
st.set_page_config(page_title="Asisten Komunitas AI Pro", page_icon="🤖", layout="wide")

st.title("🤖 Asisten Pintar & Penyederhana Informasi Komunitas")
st.write("Aplikasi Pintar Akses Informasi Valid, Pendeteksi Hoaks, dan Penyederhana Regulasi Panjang.")
st.markdown("---")

tab1, tab2 = st.tabs(["💬 Asisten Tanya Jawab & Anti-Hoaks", "📄 Penyederhana Dokumen Resmi (Simplifier)"])

# ---------------------------------------------------------------------
# TAB 1: LOGIKA CHATBOT
# ---------------------------------------------------------------------
with tab1:
    st.subheader("Tanyakan Prosedur Layanan atau Verifikasi Klaim")
    st.caption("Contoh: 'bagaimana cara daftar bansos?' atau 'pendaftaran pmi lewat wa apakah asli?'")
    
    teks_masukan = st.text_input("Masukkan pertanyaan Anda:", placeholder="Ketik pertanyaan di sini...", key="input_faq")
    
    if teks_masukan:
        with st.spinner("Menganalisis basis pengetahuan valid..."):
            jawaban, sumber, topik, akurasi = proses_pencarian_ai(teks_masukan, df, vectorizer_faq, matriks_faq)
            
            if jawaban:
                st.markdown(f"#### 📋 Hasil Analisis AI (Kategori: {topik})")
                
                if "HOAKS ALERT!" in jawaban:
                    st.error(jawaban)
                else:
                    st.success(jawaban)
                    
                st.info(f"🌐 **Sumber Valid Terverifikasi:** [{sumber}](https://{sumber})")
                st.caption(f"Tingkat kemiripan model: {akurasi*100:.2f}%")
            else:
                st.warning("⚠️ Informasi tidak ditemukan di sistem lokal atau tingkat relevansi terlalu rendah. Tetap waspada terhadap hoaks.")

# ---------------------------------------------------------------------
# TAB 2: TEXT SIMPLIFIER
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Penyederhana Dokumen Birokrasi")
    st.write("Tempel regulasi/pengumuman panjang untuk mengekstrak kalimat-kalimat inti yang paling penting secara otomatis.")
    
    contoh_teks = (
        "Berdasarkan Surat Keputusan Bersama mengenai standarisasi penyaluran jaminan sosial nasional, "
        "setiap warga masyarakat diwajibkan untuk terlebih dahulu melakukan registrasi pada sistem DTKS kementerian. "
        "Prosedur ini melibatkan validasi berlapis dimulai dari tingkat RT hingga kelurahan setempat. "
        "Setelah data terverifikasi di kelurahan, barulah berkas fisik berupa KTP dan KK asli dikirimkan ke Dinas Sosial. "
        "Masyarakat dilarang keras memberikan imbalan uang kepada petugas lapangan selama proses penyaluran berlangsung."
    )
    
    teks_panjang_input = st.text_area("Tempel teks panjang di sini:", value=contoh_teks, height=150, key="input_simplifier")
    jumlah_output = st.slider("Jumlah kalimat ringkasan:", min_value=1, max_value=3, value=2)
    
    if st.button("Sederhanakan Teks Sekarang", key="btn_simplifier"):
        with st.spinner("Mengekstrak poin penting..."):
            hasil_ringkas = sederhanakan_teks(teks_panjang_input, jumlah_kalimat=jumlah_output)
            st.markdown("#### 🎯 Hasil Ringkasan Poin Inti:")
            st.success(hasil_ringkas)
            st.caption("ℹ️ *Fitur ini menggunakan algoritma Extractive Summarization berbasis TF-IDF untuk mendeteksi kalimat kunci.*")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Ekshibisi KA/AI - LKS Dikmen Tingkat Nasional 2026</p>", unsafe_allow_html=True)
