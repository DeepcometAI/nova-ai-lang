"""
Minimal working demo for NOVA stdlib (Python-backed).

Run:
  python STDLIB_DEMO.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure stdlib/ is importable when running from repo root.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "stdlib"))


def main() -> None:
    from cosmos.stats import pearson, linear_fit
    from cosmos.orbital import delta_v, gravitational_parameter, hohmann_delta_v
    from cosmos.spectral import blackbody_peak_wavelength, doppler_shift
    from cosmos.plot import scatter, regression_line, set_title, save
    from nova.crypto import sha256
    from nova.db import connect_sqlite, execute, query_all

    # --- cosmos.stats ---
    import numpy as np

    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 6, 8, 10], dtype=float)
    r = pearson(x, y)
    slope, intercept = linear_fit(x, y)

    # --- cosmos.orbital ---
    dv = delta_v(isp_s=311.0, m_wet_kg=549054.0, m_dry_kg=25600.0)
    mu_earth = gravitational_parameter(5.972e24)
    dv1, dv2, dv_total = hohmann_delta_v(r1_m=6.771e6, r2_m=4.2164e7, mu_m3_s2=mu_earth)

    # --- cosmos.spectral ---
    peak = blackbody_peak_wavelength(5772.0)  # Sun-ish surface temp
    shifted = doppler_shift(656.28e-9, radial_velocity_m_s=30_000.0)  # H-alpha-ish

    # --- nova.crypto ---
    digest = sha256(b"nova-ai-lang")

    # --- nova.db ---
    conn = connect_sqlite()
    execute(conn, "create table demo(k text, v real)")
    execute(conn, "insert into demo(k, v) values (?, ?)", ("pearson_r", r))
    execute(conn, "insert into demo(k, v) values (?, ?)", ("delta_v_m_s", dv))
    rows = query_all(conn, "select k, v from demo order by k asc")

    print("=== NOVA stdlib demo (Python-backed) ===")
    print(f"pearson(x,y) = {r:.6f}")
    print(f"linear_fit(x,y) = slope={slope:.3f}, intercept={intercept:.3f}")
    print(f"delta_v = {dv:.2f} m/s")
    print(f"earth mu = {mu_earth:.3e} m^3/s^2")
    print(f"hohmann dv1={dv1:.2f} dv2={dv2:.2f} total={dv_total:.2f} m/s")
    print(f"blackbody_peak_wavelength(5772K) = {peak:.3e} m")
    # ASCII-only output for Windows consoles with legacy code pages.
    print(f"doppler_shift(H-alpha, 30km/s) = {shifted:.3e} m")
    print(f"sha256('nova-ai-lang') = {digest}")
    print("sqlite rows:", rows)

    # --- cosmos.plot ---
    scatter(x, y, title="Demo: y vs x", xlabel="x", ylabel="y")
    regression_line(slope, intercept, x_range=(float(x.min()), float(x.max())))
    set_title("Demo: linear_fit overlay")
    save(str(ROOT / "stdlib_demo_plot.png"))
    print("wrote plot:", str(ROOT / "stdlib_demo_plot.png"))


if __name__ == "__main__":
    main()

