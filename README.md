# Dapuria

Program skripsi rekomendasi resep berdasarkan bahan yang tersedia. Aplikasi menggunakan Streamlit, retrieval SBERT/ChromaDB, dan reranking WLC dengan bobot semantic 0,9 serta efisiensi bahan 0,1. Ollama/Llama 3 bersifat opsional untuk narasi dan klarifikasi.

## Komponen program

```text
Dapuria/
├── app.py                     # Entry point Streamlit
├── src/dapuria/               # Source, UI, dan aset aplikasi
├── data/
│   ├── corpus/recipes.csv
│   └── chroma_db/             # Indeks 5.000 resep beserta file pendamping
├── scripts/rebuild_index.py   # Utilitas pembangunan ulang indeks
├── .streamlit/config.toml
├── .env.example
├── pyproject.toml
├── requirements.txt
├── run_app_windows.bat
└── run_app.sh
```

## Instalasi dan menjalankan

Gunakan Python 3.11 atau 3.12. Jalankan instalasi dari root project dalam virtual environment baru. Jangan menyalin `.venv` dari komputer atau lokasi lain.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\run_app_windows.bat
```

macOS/Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .
bash run_app.sh
```

Buka alamat yang ditampilkan Streamlit, biasanya `http://localhost:8501`. Alternatif startup dengan interpreter environment yang sudah aktif adalah `python -m streamlit run app.py`. Contoh input: `aku punya nasi sama telur` atau `telur, tahu, bawang putih`.

## Model dan konfigurasi

Model `LazarusNLP/all-indo-e5-small-v4` diunduh dari Hugging Face pada pencarian pertama jika belum tersedia dalam cache. Penggunaan pertama memerlukan internet dan ruang penyimpanan model. Untuk penggunaan tanpa internet, sediakan model SBERT lokal lengkap dan atur `DAPURIA_MODEL_NAME` ke folder model tersebut sebelum menjalankan aplikasi.

`.env.example` berisi konfigurasi default. Aplikasi membaca environment proses dan **tidak memuat file `.env` secara otomatis**. Contoh override PowerShell:

```powershell
$env:DAPURIA_HISTORY_PATH = 'D:\dapuria-data\chats.sqlite3'
.\run_app_windows.bat
```

Pada macOS/Linux, gunakan `export DAPURIA_HISTORY_PATH=/lokasi/data/chats.sqlite3` sebelum startup. Lokasi indeks dan corpus dapat dioverride melalui `DAPURIA_CHROMA_DIR` dan `DAPURIA_CORPUS_PATH`. Gunakan path absolut atau jalankan dari root project jika memakai path relatif. Default dan algoritma rekomendasi tidak perlu diubah untuk menjalankan program.

Untuk mengaktifkan Ollama, siapkan layanannya melalui `ollama serve` dan unduh model dengan `ollama pull llama3`. Default endpoint adalah `http://localhost:11434`; modelnya `llama3`. Jika Ollama tidak tersedia, aplikasi menggunakan fallback deterministik.

## Data dan penyimpanan

Pertahankan seluruh `data/chroma_db/`, corpus CSV, CSS, dan kedua gambar dalam `src/dapuria/assets/image/`. Corpus juga digunakan untuk memulihkan langkah memasak lengkap dari indeks lama. Wheel paket Python saja tidak menyertakan `app.py` dan data resep; jalankan program dari source project lengkap.

Riwayat dibuat otomatis di `data/local_history/chats.sqlite3` atau lokasi `DAPURIA_HISTORY_PATH`. Direktori tersebut perlu dapat ditulis. Riwayat dipakai bersama oleh browser yang mengakses server yang sama; program ditujukan untuk pemakaian lokal dan belum memiliki autentikasi akun.

Indeks sudah disertakan, sehingga pembangunan ulang tidak diperlukan untuk penggunaan biasa. Bila dibutuhkan, lihat opsi dengan `python scripts/rebuild_index.py --help` menggunakan interpreter environment yang sama. Opsi `--force` menghapus indeks yang ada sebelum membangunnya kembali.

Paket pengumpulan hanya memuat program dan data resep. `.venv`, cache model eksternal, credential, serta database percakapan pribadi tidak disertakan; dependency dan model disiapkan pada komputer yang menjalankan program.
