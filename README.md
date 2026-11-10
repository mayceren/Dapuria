# Dapuria

Program skripsi rekomendasi resep berdasarkan bahan yang tersedia. Aplikasi menggunakan Streamlit, retrieval SBERT/ChromaDB, dan reranking WLC dengan bobot semantic 0,9 serta efisiensi bahan 0,1. Ollama/Llama 3 bersifat opsional untuk narasi dan klarifikasi.

Nama bahan dinormalisasi sebelum pencarian dan perhitungan kecocokan: jumlah di depan/belakang bahan serta catatan pemotongan dipisahkan dari identitas bahan, alias seperti `baput`/`bamer` dikenali, dan entri angka tunggal diabaikan. Nomor daftar pada judul resep juga dihilangkan saat metadata dibaca; corpus dan indeks sumber tidak ditulis ulang.

Bawang putih segar dan bawang putih bubuk dihitung sebagai bahan berbeda, termasuk penulisan `garlic powder` dan `bawang putih (bubuk)`. Bahan `gula` dianggap setara dengan `gula pasir` hanya saat pencocokan; nama `gula` pada persediaan dan resep tetap dipertahankan. Gula merah, gula aren, gula jawa, gula palem, serta gula halus tidak otomatis dianggap tersedia dari input `gula`.

Jika pengguna menyebut `kecap` tanpa jenisnya, aplikasi menanyakan kecap manis, kecap asin, atau jenis lain sebelum mencari resep. Jawaban singkat `manis`, `asin`, dan `dua-duanya` dikenali; jenis lain dapat disebutkan langsung, misalnya `kecap ikan`. Jenis yang dipilih menggantikan entri `kecap` yang umum. Pertanyaan tidak diulang jika pengguna sudah menyebut jenisnya sejak awal, dan pilihan yang belum selesai ikut tersimpan dalam riwayat.

Input `bawang` juga diperjelas sebelum pencarian: bawang putih, bawang merah, bawang bombay, atau jenis lain yang disebutkan pengguna. Jawaban singkat seperti `putih` dan `merah` dikenali sesuai pertanyaan yang sedang ditampilkan. Jika bawang dan kecap sama-sama belum jelas, aplikasi menanyakannya satu per satu.

Jawaban seperti `ada dongg` mengonfirmasi bahan pelengkap yang sedang ditanyakan tanpa menambah kata percakapan sebagai bahan. Pernyataan `ga punya tempe, telur toge` menghapus bahan tersebut dari persediaan dan langsung mencari ulang resep tanpa tempe, telur, atau tauge. Batasan berlaku selama percakapan sampai diubah dengan `boleh pakai tauge` atau `hapus batasan`. Nama `toge`/`taoge` disamakan dengan `tauge`; `micin`, `vetsin`, `MSG`, serta penyebutan Sasa/Ajinomoto sebagai penyedap disamakan dengan `msg`. Produk lain seperti `santan Sasa` dan penyedap berlabel non-MSG tetap dibedakan.

Setelah perhitungan WLC, kandidat harus memenuhi ambang relevansi, memiliki bahan utama yang cocok dengan persediaan, dan lolos batasan pengguna. Pemilihan hasil akhir mengutamakan jumlah bahan yang masih kurang paling sedikit, lalu efisiensi bahan dan skor WLC sebagai pembeda. Variasi judul masakan yang sangat mirip disaring agar alternatif lebih beragam; jumlah hasil mengikuti kandidat berbeda yang layak. Urutan hasil akhir dapat berbeda dari urutan skor WLC murni, sehingga evaluasi versi sebelumnya perlu diperbarui untuk menilai perilaku ini.

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
$env:DAPURIA_HISTORY_BACKEND = 'sqlite'
$env:DAPURIA_HISTORY_PATH = 'D:\dapuria-data\chats.sqlite3'
.\run_app_windows.bat
```

Pada macOS/Linux, gunakan `export DAPURIA_HISTORY_BACKEND=sqlite` dan `export DAPURIA_HISTORY_PATH=/lokasi/data/chats.sqlite3` sebelum startup untuk memakai SQLite. Lokasi indeks dan corpus dapat dioverride melalui `DAPURIA_CHROMA_DIR` dan `DAPURIA_CORPUS_PATH`. Gunakan path absolut atau jalankan dari root project jika memakai path relatif. Default dan algoritma rekomendasi tidak perlu diubah untuk menjalankan program.

Untuk mengaktifkan Ollama, siapkan layanannya melalui `ollama serve` dan unduh model dengan `ollama pull llama3`. Default endpoint adalah `http://localhost:11434`; modelnya `llama3`. Jika Ollama tidak tersedia, aplikasi menggunakan fallback deterministik.

## Data dan penyimpanan

Pertahankan seluruh `data/chroma_db/`, corpus CSV, CSS, JavaScript penyimpanan browser, dan kedua gambar dalam `src/dapuria/assets/image/`. Corpus juga digunakan untuk memulihkan langkah memasak lengkap dari indeks lama. Wheel paket Python saja tidak menyertakan `app.py` dan data resep; jalankan program dari source project lengkap.

Riwayat secara default disimpan dalam **IndexedDB browser pengguna** (`DAPURIA_HISTORY_BACKEND=browser`), termasuk saat aplikasi berjalan di Streamlit Community Cloud. Setelah server tidur, reboot, atau deploy ulang, aplikasi memuat kembali percakapan dari browser yang sama. Riwayat mencakup pesan, rekomendasi beserta detail resep, judul, pin, persediaan, batasan bahan, dan konfirmasi yang belum selesai. Penulisan ditandai selesai setelah browser mengonfirmasi transaksi; jika gagal, aplikasi menampilkan pemberitahuan dan menyediakan percobaan ulang. Perubahan yang berbenturan dari dua tab disimpan sebagai salinan.

Data tidak tersinkronisasi antarperangkat; penghapusan data situs, mode privat, kebijakan pembersihan browser, atau pergantian domain dapat membuat riwayat tidak tersedia. Profil browser berbeda memiliki riwayat terpisah; aplikasi belum memiliki autentikasi akun.

Penyimpanan SQLite lama tetap tersedia dengan `DAPURIA_HISTORY_BACKEND=sqlite`; lokasinya `data/local_history/chats.sqlite3` atau `DAPURIA_HISTORY_PATH`. Ini menyimpan di perangkat server dan membutuhkan direktori yang dapat ditulis. Mode SQLite memakai riwayat bersama untuk semua pengunjung server, sehingga gunakan mode browser untuk deployment Cloud tanpa login. Database lama tidak dihapus atau dipindahkan otomatis ke browser pengunjung.

Indeks sudah disertakan, sehingga pembangunan ulang tidak diperlukan untuk penggunaan biasa. Bila dibutuhkan, lihat opsi dengan `python scripts/rebuild_index.py --help` menggunakan interpreter environment yang sama. Opsi `--force` menghapus indeks yang ada sebelum membangunnya kembali.

Paket pengumpulan hanya memuat program dan data resep. `.venv`, cache model eksternal, credential, serta database percakapan pribadi tidak disertakan; dependency dan model disiapkan pada komputer yang menjalankan program.

Pemeriksaan regresi pengembangan tersedia melalui `python -m pytest tests -q` jika `pytest` terpasang. Test tidak memuat model SBERT dan tidak membangun ulang indeks.

Uji IndexedDB langsung di Chromium tersedia melalui `python tests/browser_history_smoke.py` menggunakan interpreter yang memiliki Playwright beserta browser Chromium. Uji menjalankan server sementara dengan retriever tiruan, menyimulasikan restart proses, menguji isolasi antarbrowser dan kegagalan penyimpanan, lalu menghentikan server uji. Dependency Playwright hanya untuk pengembangan dan tidak diperlukan pada deployment aplikasi.
