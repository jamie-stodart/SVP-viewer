"""Subset WOA23 monthly T/S for northern Australia into small Int16 files.

Run from the repo root:  python scripts/build_data.py [all|recent|both]
  all    = 1955-2022 average (decav)  -> data/
  recent = 2015-2022 decade (B5C2)    -> data/recent/
Needs: pip install numpy xarray netCDF4
Output: data/meta.json and data/m01.bin ... data/m12.bin

Reads NOAA's OPeNDAP endpoint, so only the requested region is transferred
(a few MB per file) instead of the full global NetCDFs.

Per-month layout (Int16, little-endian):
  [temperature block][salinity block], each ordered [lat][lon][depth]
  temperature = value * 100 (degC), salinity = value * 1000 (PSU), -32768 = no data
"""
import json, pathlib, sys
import numpy as np
import xarray as xr

LAT = (-45, -5)       # south, north
LON = (105, 160)      # west, east
MAX_DEPTH = 1000      # m
STEP = 2              # 2 = every 2nd cell of the 0.25 deg grid (0.5 deg). Use 1 for full res (~4x bigger)
URL = ("https://www.ncei.noaa.gov/thredds-ocean/dodsC/woa23/DATA/"
       "{v}/netcdf/{p}/0.25/woa23_{p}_{k}{m:02d}_04.nc")
SETS = {"all": ("decav", "data"), "recent": ("B5C2", "data/recent")}

def fetch(v, k, m, period):
    url = URL.format(v=v, k=k, m=m, p=period)
    print("reading", url)
    ds = xr.open_dataset(url, decode_times=False)
    da = ds[f"{k}_an"].squeeze("time", drop=True)
    da = da.sel(lat=slice(*LAT), lon=slice(*LON), depth=slice(0, MAX_DEPTH))
    return da.isel(lat=slice(None, None, STEP), lon=slice(None, None, STEP)).transpose("lat", "lon", "depth").load()

def pack(da, scale):
    a = np.round(da.values * scale)
    a[np.isnan(da.values)] = -32768
    return a.astype("<i2")

def build(period, outdir):
    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for m in range(1, 13):
        t, s = fetch("temperature", "t", m, period), fetch("salinity", "s", m, period)
        (out / f"m{m:02d}.bin").write_bytes(pack(t, 100).tobytes() + pack(s, 1000).tobytes())
        if m == 1:
            meta = dict(lon0=float(t.lon[0]), lat0=float(t.lat[0]), res=float(t.lon[1] - t.lon[0]),
                        nx=t.sizes["lon"], ny=t.sizes["lat"], depths=[float(d) for d in t.depth])
            (out / "meta.json").write_text(json.dumps(meta))
            print(outdir, "grid", meta["nx"], "x", meta["ny"], "x", len(meta["depths"]), "levels")

arg = sys.argv[1] if len(sys.argv) > 1 else "all"
for name in (SETS if arg == "both" else [arg]):
    build(*SETS[name])
print("done")
