import json
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from ftplib import FTP, error_perm
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image

FTP_SERVER = "ftp.ptree.jaxa.jp"
AHI_SUFFIX = "02801_02401"
HOTSPOT_SUFFIX = "06001_06001"
LOOKBACK_HOURS = 6
HOTSPOT_FALLBACK_HOURS = 2
ARCHIVE_RETENTION_DAYS = 7

WORK_DIR = Path("work")
AHI_DIR = WORK_DIR / "AHI"
HOTSPOT_DIR = WORK_DIR / "HOTSPOT"
PRODUCT_DIR = Path("products")
LATEST_DIR = PRODUCT_DIR / "latest"
ARCHIVE_DIR = PRODUCT_DIR / "archive"

AREA_PRESETS = {
    "Indonesia": {"lon_min": 94.5, "lon_max": 141.5, "lat_min": -11.5, "lat_max": 8.0},
    "Berau": {"lon_min": 116.0, "lon_max": 119.5, "lat_min": 0.5, "lat_max": 2.8},
}

SMOKE_RGB = {
    "r_min": 0.0, "r_max": 110.0, "r_gamma": 1.8,
    "g_min": 0.0, "g_max": 100.0, "g_gamma": 1.0,
    "b_min": 8.0, "b_max": 60.0, "b_gamma": 3.0,
}

AHI_PATTERN = re.compile(r"NC_(H\d{2})_(\d{8})_(\d{4})_R21_FLDK\.(\d{5}_\d{5})\.nc$")
HOTSPOT_PATTERN = re.compile(r"H\d{2}_(\d{8})_(\d{4})_L2WLF010_FLDK\.(\d{5}_\d{5})\.csv$")


def parse_ahi_time(path):
    match = AHI_PATTERN.match(Path(path).name)
    return None if match is None else datetime.strptime(match.group(2) + match.group(3), "%Y%m%d%H%M")


def parse_hotspot_time(path):
    match = HOTSPOT_PATTERN.match(Path(path).name)
    return None if match is None else datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M")


def connect_ftp(username, password, timeout=180):
    ftp = FTP(FTP_SERVER, timeout=timeout)
    ftp.login(user=username, passwd=password)
    return ftp


def safe_ftp_close(ftp):
    if ftp is None:
        return
    try:
        ftp.quit()
    except Exception:
        try:
            ftp.close()
        except Exception:
            pass


def ahi_remote_directory(timestamp):
    return f"/jma/netcdf/{timestamp:%Y%m}/{timestamp:%d}/"


def hotspot_remote_directory(timestamp):
    return f"/pub/himawari/L2/WLF/010/{timestamp:%Y%m}/{timestamp:%d}/{timestamp:%H}/"


def list_remote_names(ftp, remote_directory):
    ftp.cwd(remote_directory)
    return [Path(name).name for name in ftp.nlst()]


def remote_file_size(ftp, filename):
    try:
        size = ftp.size(filename)
        return int(size) if size is not None else None
    except Exception:
        return None


def download_ftp_file(ftp, filename, local_path):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    expected_size = remote_file_size(ftp, filename)

    if local_path.exists() and (expected_size is None or local_path.stat().st_size == expected_size):
        return local_path, "skipped"

    partial = local_path.with_suffix(local_path.suffix + ".part")
    partial.unlink(missing_ok=True)

    with partial.open("wb") as output:
        ftp.retrbinary(f"RETR {filename}", output.write, blocksize=1024 * 1024)

    if expected_size is not None and partial.stat().st_size != expected_size:
        partial.unlink(missing_ok=True)
        raise IOError("Ukuran file hasil download tidak cocok dengan ukuran FTP.")

    partial.replace(local_path)
    return local_path, "downloaded"


def latest_remote_ahi(ftp, now_utc):
    cutoff = now_utc - timedelta(hours=LOOKBACK_HOURS)
    candidates = []
    for target_date in sorted({now_utc.date(), cutoff.date()}):
        probe = datetime.combine(target_date, datetime.min.time())
        remote_dir = ahi_remote_directory(probe)
        try:
            names = list_remote_names(ftp, remote_dir)
        except error_perm:
            continue
        for name in names:
            timestamp = parse_ahi_time(name)
            if timestamp and name.endswith(f"{AHI_SUFFIX}.nc") and cutoff <= timestamp <= now_utc:
                candidates.append((timestamp, remote_dir, name))
    return None if not candidates else max(candidates, key=lambda item: item[0])


def best_remote_hotspot(ftp, ahi_time):
    earliest = ahi_time - timedelta(hours=HOTSPOT_FALLBACK_HOURS)
    cursor = earliest.replace(minute=0, second=0, microsecond=0)
    last_hour = ahi_time.replace(minute=0, second=0, microsecond=0)
    candidates = []
    while cursor <= last_hour:
        remote_dir = hotspot_remote_directory(cursor)
        try:
            names = list_remote_names(ftp, remote_dir)
        except error_perm:
            cursor += timedelta(hours=1)
            continue
        for name in names:
            timestamp = parse_hotspot_time(name)
            if timestamp and name.endswith(f"{HOTSPOT_SUFFIX}.csv") and earliest <= timestamp <= ahi_time:
                candidates.append((timestamp, remote_dir, name))
        cursor += timedelta(hours=1)
    return None if not candidates else max(candidates, key=lambda item: item[0])


def find_name(dataset, candidates):
    available = set(dataset.coords) | set(dataset.variables)
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise KeyError(f"Tidak menemukan koordinat: {candidates}")


def normalize(data, vmin, vmax, gamma):
    channel = (np.asarray(data, dtype=np.float32) - vmin) / (vmax - vmin)
    return np.clip(channel, 0.0, 1.0) ** (1.0 / gamma)


def read_ahi_crop(nc_path, area):
    with xr.open_dataset(nc_path, engine="h5netcdf", decode_timedelta=False, cache=False) as ds:
        lon_name = find_name(ds, ["longitude", "lon"])
        lat_name = find_name(ds, ["latitude", "lat"])
        lat_desc = float(ds[lat_name].values[0]) > float(ds[lat_name].values[-1])
        crop = ds.sel({
            lon_name: slice(area["lon_min"], area["lon_max"]),
            lat_name: slice(area["lat_max"], area["lat_min"]) if lat_desc else slice(area["lat_min"], area["lat_max"]),
        })
        lon = np.asarray(crop[lon_name].values, dtype=np.float32)
        lat = np.asarray(crop[lat_name].values, dtype=np.float32)
        b03 = np.asarray(crop["albedo_03"].squeeze().values, dtype=np.float32)
        b04 = np.asarray(crop["albedo_04"].squeeze().values, dtype=np.float32)
        b06 = np.asarray(crop["albedo_06"].squeeze().values, dtype=np.float32)
    if np.nanpercentile(b03, 99.9) <= 2.0:
        b03 *= 100.0; b04 *= 100.0; b06 *= 100.0
    return lon, lat, b03, b04, b06


def make_smoke_rgb(b03, b04, b06):
    red = normalize(b03, SMOKE_RGB["r_min"], SMOKE_RGB["r_max"], SMOKE_RGB["r_gamma"])
    green = normalize(b04, SMOKE_RGB["g_min"], SMOKE_RGB["g_max"], SMOKE_RGB["g_gamma"])
    blue = normalize(b06, SMOKE_RGB["b_min"], SMOKE_RGB["b_max"], SMOKE_RGB["b_gamma"])
    return np.clip(np.nan_to_num(np.dstack([red, green, blue]), nan=0.0), 0.0, 1.0)


def read_hotspot(csv_path):
    if csv_path is None:
        return pd.DataFrame()
    hotspot = pd.read_csv(csv_path, skiprows=1)
    hotspot.columns = [str(c).replace("# ", "").strip() for c in hotspot.columns]
    for column in ["Lat", "Lon", "Area(km^2)", "Volcano", "Level", "Reliability", "FRP(Wm^-2)", "QF", "Hot(ID)"]:
        if column in hotspot.columns:
            hotspot[column] = pd.to_numeric(hotspot[column], errors="coerce")
    if {"Lat", "Lon"}.issubset(hotspot.columns):
        hotspot = hotspot.dropna(subset=["Lat", "Lon"])
    return hotspot.reset_index(drop=True)


def save_raw_rgb(rgb, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(output_path, optimize=True)


def render_archive_png(rgb, lon, lat, hotspot, area_name, area, observation_time):
    fig, ax = plt.subplots(figsize=(16, 9))
    origin = "upper" if lat[0] > lat[-1] else "lower"
    ax.imshow(rgb, extent=[float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())], origin=origin, interpolation="nearest")
    hs = hotspot.copy()
    if not hs.empty:
        hs = hs[(hs["Lon"] >= area["lon_min"]) & (hs["Lon"] <= area["lon_max"]) & (hs["Lat"] >= area["lat_min"]) & (hs["Lat"] <= area["lat_max"])]
    if not hs.empty:
        ax.scatter(hs["Lon"], hs["Lat"], s=35, facecolors="red", edgecolors="yellow", linewidths=0.8, alpha=0.93, zorder=20, label=f"Hotspot (n={len(hs)})")
        ax.legend(loc="lower right", framealpha=0.85)
    ax.set_xlim(area["lon_min"], area["lon_max"]); ax.set_ylim(area["lat_min"], area["lat_max"])
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(alpha=0.25)
    ax.set_title(f"{area_name} Fire & Smoke Monitor\n{observation_time:%d %B %Y, %H:%M UTC}", fontweight="bold")
    ax.text(0.012, 0.018, "Interpretasi cepat\nKuning/cokelat samar: indikasi asap\nPutih terang: awan\nTitik merah-kuning: hotspot", transform=ax.transAxes, color="white", fontsize=9, va="bottom", bbox={"facecolor": "black", "edgecolor": "white", "alpha": 0.68, "pad": 5})
    fig.tight_layout()
    out_dir = ARCHIVE_DIR / area_name / observation_time.strftime("%Y") / observation_time.strftime("%m") / observation_time.strftime("%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"FireSmoke_{area_name}_{observation_time:%Y%m%d_%H%M}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def remove_old_archives():
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=ARCHIVE_RETENTION_DAYS)
    removed = 0
    for path in ARCHIVE_DIR.rglob("*.png"):
        match = re.search(r"_(\d{8})_(\d{4})\.png$", path.name)
        if match and datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M") < cutoff:
            path.unlink(missing_ok=True); removed += 1
    for directory in sorted([p for p in ARCHIVE_DIR.rglob("*") if p.is_dir()], reverse=True):
        try: directory.rmdir()
        except OSError: pass
    return removed


def main():
    username = os.environ.get("PTREE_USERNAME", "")
    password = os.environ.get("PTREE_PASSWORD", "")
    if not username or not password:
        raise RuntimeError("PTREE_USERNAME/PTREE_PASSWORD belum tersedia di GitHub Secrets.")

    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    AHI_DIR.mkdir(parents=True, exist_ok=True)
    HOTSPOT_DIR.mkdir(parents=True, exist_ok=True)
    ftp = None
    try:
        ftp = connect_ftp(username, password)
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        ahi_remote = latest_remote_ahi(ftp, now_utc)
        if ahi_remote is None:
            raise RuntimeError("AHI terbaru tidak ditemukan.")
        ahi_time, ahi_remote_dir, ahi_name = ahi_remote
        ftp.cwd(ahi_remote_dir)
        ahi_local, ahi_result = download_ftp_file(ftp, ahi_name, AHI_DIR / ahi_name)

        hotspot_remote = best_remote_hotspot(ftp, ahi_time)
        hotspot_local = None; hotspot_time = None; hotspot_result = "not available"
        if hotspot_remote is not None:
            hotspot_time, hotspot_remote_dir, hotspot_name = hotspot_remote
            ftp.cwd(hotspot_remote_dir)
            hotspot_local, hotspot_result = download_ftp_file(ftp, hotspot_name, HOTSPOT_DIR / hotspot_name)

        hotspot = read_hotspot(hotspot_local)
        hotspot.to_csv(LATEST_DIR / "latest_hotspot.csv", index=False)
        area_metadata = {}

        for area_name, area in AREA_PRESETS.items():
            lon, lat, b03, b04, b06 = read_ahi_crop(ahi_local, area)
            rgb = make_smoke_rgb(b03, b04, b06)
            raw_name = f"{area_name.lower()}_rgb.png"
            save_raw_rgb(rgb, LATEST_DIR / raw_name)
            archived = render_archive_png(rgb, lon, lat, hotspot, area_name, area, ahi_time)
            latest_png = LATEST_DIR / f"latest_{area_name.lower()}.png"
            shutil.copy2(archived, latest_png)
            area_metadata[area_name] = {
                "rgb_file": raw_name,
                "latest_png": latest_png.name,
                "lon_min": float(lon.min()), "lon_max": float(lon.max()),
                "lat_min": float(lat.min()), "lat_max": float(lat.max()),
                "width": int(len(lon)), "height": int(len(lat)),
            }

        metadata = {
            "generated_at_utc": now_utc.isoformat() + "Z",
            "observation_time_utc": ahi_time.isoformat() + "Z",
            "hotspot_time_utc": hotspot_time.isoformat() + "Z" if hotspot_time else None,
            "ahi_file": ahi_name,
            "ahi_download_result": ahi_result,
            "hotspot_download_result": hotspot_result,
            "archive_retention_days": ARCHIVE_RETENTION_DAYS,
            "removed_archives": remove_old_archives(),
            "areas": area_metadata,
        }
        (LATEST_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2))
    finally:
        safe_ftp_close(ftp)
        shutil.rmtree(WORK_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
