import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Indonesia Fire & Smoke Monitor", page_icon="🔥", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 0.7rem; padding-bottom: 1rem; }
.main-header { background: linear-gradient(90deg,#7f1d1d 0%,#991b1b 45%,#b91c1c 100%); border-radius:14px; padding:.9rem 1.1rem; color:white; margin-bottom:.7rem; }
.main-header h1 { margin:0; font-size:1.65rem; }
.main-header p { margin:.25rem 0 0 0; font-size:.88rem; }
</style>
<div class="main-header">
<h1>Indonesia Fire & Smoke Monitor</h1>
<p>Monitoring hotspot dan indikasi asap berbasis Himawari Smoke RGB.</p>
<p>Indonesia Coverage dan Area Khusus Pemantauan Kab. Berau</p>
<p>Created by ulil.hidayat@bmkg.go.id & Tim BMKG Berau</p>
</div>
""", unsafe_allow_html=True)

LATEST_DIR = Path("products/latest")
ARCHIVE_DIR = Path("products/archive")
METADATA_PATH = LATEST_DIR / "metadata.json"
HOTSPOT_PATH = LATEST_DIR / "latest_hotspot.csv"


def load_metadata():
    return None if not METADATA_PATH.exists() else json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def load_hotspot():
    if not HOTSPOT_PATH.exists():
        return pd.DataFrame()
    hotspot = pd.read_csv(HOTSPOT_PATH)
    for column in ["Lat", "Lon", "Reliability", "FRP(Wm^-2)"]:
        if column in hotspot.columns:
            hotspot[column] = pd.to_numeric(hotspot[column], errors="coerce")
    return hotspot.dropna(subset=["Lat", "Lon"])


def make_map(rgb_path, area_meta, hotspot, minimum_reliability, marker_size):
    image = np.asarray(Image.open(rgb_path).convert("RGB"))
    lon_min, lon_max = area_meta["lon_min"], area_meta["lon_max"]
    lat_min, lat_max = area_meta["lat_min"], area_meta["lat_max"]
    dx = (lon_max - lon_min) / max(image.shape[1] - 1, 1)
    dy = -(lat_max - lat_min) / max(image.shape[0] - 1, 1)

    fig = go.Figure()
    fig.add_trace(go.Image(z=image, x0=lon_min, dx=dx, y0=lat_max, dy=dy, name="Smoke RGB"))

    hs = hotspot.copy()
    if "Reliability" in hs.columns:
        hs = hs[hs["Reliability"] >= minimum_reliability]
    hs = hs[(hs["Lon"] >= lon_min) & (hs["Lon"] <= lon_max) & (hs["Lat"] >= lat_min) & (hs["Lat"] <= lat_max)]

    if not hs.empty:
        hover = []
        for _, row in hs.iterrows():
            lines = [f"Lat: {row['Lat']:.2f}", f"Lon: {row['Lon']:.2f}"]
            for col in ["Reliability", "FRP(Wm^-2)", "Area(km^2)", "Level"]:
                if col in hs.columns and pd.notna(row.get(col)):
                    lines.append(f"{col}: {row[col]}")
            hover.append("<br>".join(lines))
        fig.add_trace(go.Scatter(x=hs["Lon"], y=hs["Lat"], mode="markers", marker={"size":marker_size,"color":"red","line":{"color":"yellow","width":1},"opacity":.92}, text=hover, hovertemplate="%{text}<extra></extra>", name=f"Hotspot (n={len(hs)})"))

    fig.update_layout(height=760, margin={"l":20,"r":20,"t":40,"b":20}, dragmode="zoom", legend={"orientation":"h","yanchor":"bottom","y":1.01,"xanchor":"left","x":0})
    fig.update_xaxes(range=[lon_min, lon_max], title="Longitude", showgrid=True)
    fig.update_yaxes(range=[lat_min, lat_max], title="Latitude", scaleanchor="x", scaleratio=1, showgrid=True)
    return fig, hs


def list_archives(area_name):
    folder = ARCHIVE_DIR / area_name
    return [] if not folder.exists() else sorted(folder.rglob("*.png"), reverse=True)


metadata = load_metadata()
if metadata is None:
    st.error("Produk belum tersedia. Jalankan GitHub Actions `Update Fire Smoke Products` terlebih dahulu.")
    st.stop()

hotspot = load_hotspot()
observation_time = datetime.fromisoformat(metadata["observation_time_utc"].replace("Z", "+00:00"))
generated_time = datetime.fromisoformat(metadata["generated_at_utc"].replace("Z", "+00:00"))
age_minutes = max(int((datetime.now(timezone.utc) - observation_time).total_seconds() / 60), 0)

with st.sidebar:
    st.markdown("### Operational controls")
    selected_area = st.radio("Area", ["Indonesia", "Berau"], horizontal=True)
    minimum_reliability = st.selectbox("Hotspot confidence", [1,2,3], format_func=lambda v:{1:"Level 1–3",2:"Level 2–3",3:"Level 3"}[v])
    marker_size = st.slider("Ukuran marker", 5, 100, 10)
    if st.button("Refresh dashboard", type="primary", use_container_width=True):
        st.rerun()
    st.caption("Data diperbarui otomatis oleh GitHub Actions. Browser tidak perlu dibiarkan terbuka.")

latest_tab, archive_tab = st.tabs(["Kondisi terbaru", "Arsip 7 hari"])

with latest_tab:
    area_meta = metadata["areas"][selected_area]
    rgb_path = LATEST_DIR / area_meta["rgb_file"]
    if not rgb_path.exists():
        st.error(f"Produk RGB untuk {selected_area} belum tersedia.")
        st.stop()
    fig, area_hotspot = make_map(rgb_path, area_meta, hotspot, minimum_reliability, marker_size)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Waktu observasi", observation_time.strftime("%d %b %Y"), observation_time.strftime("%H:%M UTC"))
    c2.metric("Hotspot dalam area", len(area_hotspot), selected_area)
    c3.metric("Reliability level 3", int((area_hotspot["Reliability"] >= 3).sum()) if not area_hotspot.empty and "Reliability" in area_hotspot.columns else 0)
    c4.metric("Usia data", f"{age_minutes} menit", f"Workflow: {generated_time:%H:%M UTC}")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False})
    latest_png = LATEST_DIR / area_meta["latest_png"]
    if latest_png.exists():
        st.download_button(f"Download PNG {selected_area}", data=latest_png.read_bytes(), file_name=latest_png.name, mime="image/png")
    with st.expander("Daftar hotspot area aktif"):
        if area_hotspot.empty:
            st.info("Tidak ada hotspot sesuai filter.")
        else:
            cols = [c for c in ["Lat","Lon","Reliability","FRP(Wm^-2)","Area(km^2)","Level"] if c in area_hotspot.columns]
            st.dataframe(area_hotspot[cols], use_container_width=True, hide_index=True)

with archive_tab:
    archive_area = st.radio("Area arsip", ["Indonesia", "Berau"], horizontal=True, key="archive_area")
    archive_files = list_archives(archive_area)
    if not archive_files:
        st.info("Belum ada arsip untuk area ini.")
    else:
        selected = st.selectbox("Pilih arsip", archive_files, format_func=lambda p:p.name)
        st.image(selected.read_bytes(), caption=selected.name, use_container_width=True)
        st.download_button("Download PNG arsip", data=selected.read_bytes(), file_name=selected.name, mime="image/png")

st.divider()
st.caption("GitHub Actions memperbarui produk secara terjadwal. Arsip PNG dipertahankan selama 7 hari. NetCDF hanya digunakan sementara dan tidak disimpan di repository.")
