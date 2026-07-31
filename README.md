# Indonesia Fire & Smoke Monitor — GitHub Actions Version

Versi ini memisahkan proses otomatis dan dashboard:

- `scheduled_update.py`: download FTP JAXA, membuat produk terbaru, dan menjaga arsip 7 hari.
- `.github/workflows/update_fire_smoke.yml`: update pada menit 07 dan 37 setiap jam.
- `app.py`: membaca produk terbaru dari folder `products/`.
- NetCDF hanya dipakai sementara di runner GitHub Actions lalu dihapus.

## Setup

1. Tambahkan repository secrets `PTREE_USERNAME` dan `PTREE_PASSWORD`.
2. Buka tab Actions → Update Fire Smoke Products → Run workflow.
3. Setelah berhasil, deploy `app.py` pada Streamlit Community Cloud.

Browser tidak perlu terus terbuka agar download berjalan.
