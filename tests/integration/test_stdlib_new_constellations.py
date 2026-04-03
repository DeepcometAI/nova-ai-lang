"""
Integration tests for NOVA stdlib — new constellations (Step 12 completion)
Tests: cosmos.astro, cosmos.chem, cosmos.geo, cosmos.orbital,
       cosmos.quantum, cosmos.signal, cosmos.spectral, cosmos.thermo,
       nova.concurrent, nova.crypto, nova.db, nova.test
"""

import sys, os, math
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'stdlib'))


# ── cosmos.astro ─────────────────────────────────────────────────────────────

class TestCosmosAstro:
    def test_parallax_distance(self):
        from cosmos.astro import parallax_distance_pc
        assert abs(parallax_distance_pc(1.0) - 1.0) < 1e-10
        assert abs(parallax_distance_pc(0.5) - 2.0) < 1e-10

    def test_parallax_zero_raises(self):
        from cosmos.astro import parallax_distance_pc
        with pytest.raises(ValueError):
            parallax_distance_pc(0.0)

    def test_distance_modulus(self):
        from cosmos.astro import distance_modulus_to_pc
        # 10 pc → μ = 0.0
        d = distance_modulus_to_pc(0.0, 0.0)
        assert abs(d - 10.0) < 1e-6

    def test_bv_temperature(self):
        from cosmos.astro import bv_to_temperature_k
        # Sun: BV ≈ 0.65, T ≈ 5778 K
        t = bv_to_temperature_k(0.65)
        assert 5000 < t < 6500

    def test_spectral_class(self):
        from cosmos.astro import spectral_class
        assert spectral_class(-0.33) == "O"
        assert spectral_class(0.65)  == "G"   # Sun
        assert spectral_class(1.5)   == "M"

    def test_angular_separation_same_point(self):
        from cosmos.astro import angular_separation_deg
        sep = angular_separation_deg(10.0, 20.0, 10.0, 20.0)
        assert abs(sep) < 1e-10

    def test_angular_separation_90deg(self):
        from cosmos.astro import angular_separation_deg
        sep = angular_separation_deg(0.0, 0.0, 90.0, 0.0)
        assert abs(sep - 90.0) < 1e-6

    def test_ra_dec_roundtrip(self):
        from cosmos.astro import ra_dec_to_cartesian, cartesian_to_ra_dec
        ra, dec = 120.0, 30.0
        x, y, z = ra_dec_to_cartesian(ra, dec)
        ra2, dec2 = cartesian_to_ra_dec(x, y, z)
        assert abs(ra2 - ra) < 1e-8
        assert abs(dec2 - dec) < 1e-8

    def test_wien_displacement(self):
        from cosmos.astro import wien_displacement
        lam = wien_displacement(5778.0)  # Sun
        assert 490e-9 < lam < 520e-9     # ~501 nm

    def test_hubble_distance(self):
        from cosmos.astro import hubble_distance_mpc
        d = hubble_distance_mpc(7000.0, 70.0)
        assert abs(d - 100.0) < 1e-6


# ── cosmos.chem ──────────────────────────────────────────────────────────────

class TestCosmossChem:
    def test_element_symbol_hydrogen(self):
        from cosmos.chem import element_symbol
        assert element_symbol(1) == "H"

    def test_element_symbol_iron(self):
        from cosmos.chem import element_symbol
        assert element_symbol(26) == "Fe"

    def test_element_name(self):
        from cosmos.chem import element_name
        assert element_name(6) == "Carbon"

    def test_atomic_number(self):
        from cosmos.chem import atomic_number
        assert atomic_number("Fe") == 26
        assert atomic_number("H")  == 1

    def test_atomic_mass_hydrogen(self):
        from cosmos.chem import atomic_mass_u
        assert abs(atomic_mass_u(1) - 1.00794) < 0.001

    def test_moles_from_grams(self):
        from cosmos.chem import moles_from_grams
        # 12 g C / 12.0107 g/mol ≈ 1 mol
        n = moles_from_grams(12.0, 6)
        assert abs(n - 1.0) < 0.01

    def test_ideal_gas_volume(self):
        from cosmos.chem import ideal_gas_volume
        # 1 mol at STP (273.15K, 101325 Pa) ≈ 22.4 L
        v = ideal_gas_volume(1.0, 273.15, 101325.0)
        assert abs(v - 0.02241) < 0.001

    def test_energy_wavelength_roundtrip(self):
        from cosmos.chem import energy_to_wavelength_m, wavelength_to_energy_j
        e = 3.3e-19  # ~600 nm photon
        lam = energy_to_wavelength_m(e)
        e2  = wavelength_to_energy_j(lam)
        assert abs(e2 - e) / e < 1e-10

    def test_arrhenius_rate(self):
        from cosmos.chem import arrhenius_rate
        k = arrhenius_rate(1e13, 80_000.0, 500.0)
        assert k > 0

    def test_ionization_shells_carbon(self):
        from cosmos.chem import ionization_shells
        shells = ionization_shells(6)
        assert shells[0] == 2   # K shell
        assert shells[1] == 4   # L shell


# ── cosmos.geo ───────────────────────────────────────────────────────────────

class TestCosmosGeo:
    def test_great_circle_same_point(self):
        from cosmos.geo import great_circle_distance_m
        d = great_circle_distance_m(0.0, 0.0, 0.0, 0.0)
        assert abs(d) < 1.0

    def test_great_circle_equatorial_quarter(self):
        from cosmos.geo import great_circle_distance_m
        # Quarter of Earth's circumference ≈ 10_018 km
        d = great_circle_distance_m(0.0, 0.0, 0.0, 90.0)
        assert abs(d - 10_018_754) < 50_000

    def test_bearing_north(self):
        from cosmos.geo import bearing_deg
        b = bearing_deg(0.0, 0.0, 10.0, 0.0)
        assert abs(b) < 1.0  # heading north

    def test_bearing_east(self):
        from cosmos.geo import bearing_deg
        b = bearing_deg(0.0, 0.0, 0.0, 10.0)
        assert abs(b - 90.0) < 1.0

    def test_dms_roundtrip(self):
        from cosmos.geo import dd_to_dms, dms_to_dd
        dd = 51.5074
        d, m, s = dd_to_dms(dd)
        dd2 = dms_to_dd(d, m, s)
        assert abs(dd2 - dd) < 1e-6

    def test_geodetic_ecef_roundtrip(self):
        from cosmos.geo import geodetic_to_ecef, ecef_to_geodetic
        lat, lon, alt = 51.5, -0.1, 100.0
        x, y, z = geodetic_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_geodetic(x, y, z)
        assert abs(lat2 - lat) < 0.001
        assert abs(lon2 - lon) < 0.001
        assert abs(alt2 - alt) < 1.0

    def test_midpoint(self):
        from cosmos.geo import midpoint
        # Midpoint of equatorial segment
        lat_m, lon_m = midpoint(0.0, 0.0, 0.0, 10.0)
        assert abs(lat_m) < 0.01
        assert abs(lon_m - 5.0) < 0.01


# ── cosmos.orbital ───────────────────────────────────────────────────────────

class TestCosmosOrbital:
    def test_delta_v_falcon9(self):
        from cosmos.orbital import delta_v
        dv = delta_v(311.0, 549054.0, 25600.0)
        assert abs(dv - 9740) < 200  # ~9.74 km/s

    def test_delta_v_invalid_mass(self):
        from cosmos.orbital import delta_v
        with pytest.raises(ValueError):
            delta_v(311.0, 100.0, 200.0)  # m_wet < m_dry

    def test_kepler_period_earth(self):
        from cosmos.orbital import kepler_period
        M_SUN = 1.989e30
        AU    = 1.496e11
        T = kepler_period(AU, M_SUN)
        # Should be ~1 year = 3.156e7 s
        assert abs(T - 3.156e7) / 3.156e7 < 0.01

    def test_kepler_period_yr_earth(self):
        from cosmos.orbital import kepler_period_yr
        assert abs(kepler_period_yr(1.0) - 1.0) < 1e-10

    def test_kepler_period_yr_mars(self):
        from cosmos.orbital import kepler_period_yr
        assert abs(kepler_period_yr(1.524) - 1.881) < 0.05

    def test_escape_velocity_earth(self):
        from cosmos.orbital import escape_velocity
        M_E = 5.972e24; R_E = 6.371e6
        v = escape_velocity(R_E, M_E)
        assert abs(v - 11_186) < 100  # ~11.2 km/s

    def test_circular_velocity(self):
        from cosmos.orbital import circular_velocity, escape_velocity
        M = 5.972e24; r = 6.371e6 + 400e3  # ISS orbit
        vc = circular_velocity(r, M)
        ve = escape_velocity(r, M)
        # v_esc = √2 * v_circ
        assert abs(ve / vc - math.sqrt(2)) < 0.001

    def test_hohmann_dv_positive(self):
        from cosmos.orbital import hohmann_delta_v, gravitational_parameter
        mu = gravitational_parameter(5.972e24)
        r1 = 6.371e6 + 400e3
        r2 = 6.371e6 + 35786e3  # GEO
        dv1, dv2, total = hohmann_delta_v(r1, r2, mu)
        assert dv1 > 0 and dv2 > 0 and total > 0

    def test_orbital_energy_bound(self):
        from cosmos.orbital import orbital_energy, circular_velocity
        M = 5.972e24; r = 7e6
        vc = circular_velocity(r, M)
        e = orbital_energy(r, vc, M)
        assert e < 0  # bound orbit


# ── cosmos.quantum ───────────────────────────────────────────────────────────

class TestCosmosQuantum:
    def test_ket_zero_normalised(self):
        from cosmos.quantum import ket_zero
        psi = ket_zero()
        assert abs(np.dot(np.conj(psi), psi) - 1.0) < 1e-10

    def test_hadamard_unitary(self):
        from cosmos.quantum import hadamard
        H = hadamard()
        I = H @ H.conj().T
        assert np.allclose(I, np.eye(2), atol=1e-10)

    def test_pauli_x_flips(self):
        from cosmos.quantum import pauli_x, ket_zero, ket_one, apply_gate
        result = apply_gate(pauli_x(), ket_zero())
        assert np.allclose(result, ket_one(), atol=1e-10)

    def test_hadamard_superposition(self):
        from cosmos.quantum import hadamard, ket_zero, apply_gate, ket_plus
        result = apply_gate(hadamard(), ket_zero())
        assert np.allclose(result, ket_plus(), atol=1e-10)

    def test_measure_collapses(self):
        from cosmos.quantum import measure, ket_zero
        idx, state = measure(ket_zero())
        assert idx == 0
        assert abs(state[0]) == 1.0

    def test_inner_product_orthogonal(self):
        from cosmos.quantum import inner_product, ket_zero, ket_one
        ip = inner_product(ket_zero(), ket_one())
        assert abs(ip) < 1e-10

    def test_hydrogen_ground_state(self):
        from cosmos.quantum import hydrogen_energy_ev
        e = hydrogen_energy_ev(1)
        assert abs(e - (-13.606)) < 0.01

    def test_de_broglie_electron(self):
        from cosmos.quantum import de_broglie_wavelength
        m_e = 9.109e-31
        lam = de_broglie_wavelength(m_e, 1e6)  # 1 MeV electron approx
        assert lam > 0

    def test_tensor_product_shape(self):
        from cosmos.quantum import tensor_product, ket_zero, ket_one
        combined = tensor_product(ket_zero(), ket_one())
        assert combined.shape == (4,)


# ── cosmos.signal ────────────────────────────────────────────────────────────

class TestCosmosSignal:
    def test_fft_roundtrip(self):
        from cosmos.signal import fft, ifft
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.allclose(np.real(ifft(fft(x))), x, atol=1e-10)

    def test_fft_dc_component(self):
        from cosmos.signal import fft
        x = np.ones(8)
        F = fft(x)
        assert abs(F[0] - 8.0) < 1e-10  # DC = sum

    def test_power_spectrum_length(self):
        from cosmos.signal import power_spectrum
        x = np.random.randn(64)
        freqs, power = power_spectrum(x, sample_rate_hz=100.0)
        assert len(freqs) == 33  # N//2 + 1
        assert len(power) == 33

    def test_convolve_impulse(self):
        from cosmos.signal import convolve
        x = np.array([1.0, 2.0, 3.0])
        impulse = np.array([1.0])
        result = convolve(x, impulse, mode="full")
        assert np.allclose(result, x, atol=1e-10)

    def test_window_hann_length(self):
        from cosmos.signal import window_hann
        w = window_hann(64)
        assert len(w) == 64
        assert abs(w[0]) < 1e-10   # starts at 0
        assert abs(w[-1]) < 1e-10  # ends at 0

    def test_rms_constant(self):
        from cosmos.signal import rms
        x = np.ones(100) * 3.0
        assert abs(rms(x) - 3.0) < 1e-10

    def test_lowpass_removes_high_freq(self):
        from cosmos.signal import lowpass_filter
        sr = 1000.0
        t  = np.linspace(0, 1, int(sr), endpoint=False)
        # 10 Hz signal + 400 Hz noise
        x  = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*400*t)
        y  = lowpass_filter(x, cutoff_hz=50.0, sample_rate_hz=sr)
        # After filtering, the 400 Hz component should be much smaller
        from cosmos.signal import rfft
        orig_spec = np.abs(rfft(x))
        filt_spec = np.abs(rfft(y))
        # Energy at high freq should decrease
        high_idx = int(400 * len(y) / sr)
        assert filt_spec[high_idx] < orig_spec[high_idx] * 0.1


# ── cosmos.spectral ──────────────────────────────────────────────────────────

class TestCosmosSpectral:
    def test_wien_sun(self):
        from cosmos.spectral import blackbody_peak_wavelength
        lam = blackbody_peak_wavelength(5778.0)
        assert 490e-9 < lam < 520e-9

    def test_blackbody_radiance_positive(self):
        from cosmos.spectral import blackbody_spectral_radiance
        b = blackbody_spectral_radiance(500e-9, 5778.0)
        assert b > 0

    def test_doppler_redshift(self):
        from cosmos.spectral import doppler_shift_wavelength
        lam0 = 656.3e-9  # H-alpha
        lam1 = doppler_shift_wavelength(lam0, 1e6)  # receding at 1000 km/s
        assert lam1 > lam0  # redshifted

    def test_redshift_velocity_roundtrip(self):
        from cosmos.spectral import redshift_from_velocity, velocity_from_redshift
        v = 1e6  # 1000 km/s
        z = redshift_from_velocity(v)
        v2 = velocity_from_redshift(z)
        assert abs(v2 - v) / v < 1e-6

    def test_balmer_halpha(self):
        from cosmos.spectral import hydrogen_balmer_wavelength
        lam = hydrogen_balmer_wavelength(3)  # Hα
        assert abs(lam * 1e9 - 656.279) < 0.5  # ±0.5 nm

    def test_emission_lines_dict(self):
        from cosmos.spectral import emission_line_wavelengths
        lines = emission_line_wavelengths()
        assert "H-alpha" in lines
        assert lines["H-alpha"] > 0

    def test_luminosity_distance_positive(self):
        from cosmos.spectral import luminosity_distance_mpc
        d = luminosity_distance_mpc(0.1)
        assert d > 0

    def test_comoving_distance_increases_with_z(self):
        from cosmos.spectral import comoving_distance_mpc
        d1 = comoving_distance_mpc(0.1)
        d2 = comoving_distance_mpc(0.5)
        assert d2 > d1


# ── cosmos.thermo ────────────────────────────────────────────────────────────

class TestCosmosThermo:
    def test_ideal_gas_pressure(self):
        from cosmos.thermo import ideal_gas_pressure, ideal_gas_temperature
        P = ideal_gas_pressure(1.0, 273.15, 0.02241)
        assert abs(P - 101325) < 1000  # ~1 atm

    def test_temperature_roundtrip(self):
        from cosmos.thermo import celsius_to_kelvin, kelvin_to_celsius
        t = 25.0
        assert abs(kelvin_to_celsius(celsius_to_kelvin(t)) - t) < 1e-10

    def test_fahrenheit_roundtrip(self):
        from cosmos.thermo import fahrenheit_to_kelvin, kelvin_to_fahrenheit
        f = 72.0
        assert abs(kelvin_to_fahrenheit(fahrenheit_to_kelvin(f)) - f) < 1e-6

    def test_carnot_efficiency(self):
        from cosmos.thermo import carnot_efficiency
        eff = carnot_efficiency(600.0, 300.0)
        assert abs(eff - 0.5) < 1e-10

    def test_carnot_efficiency_range(self):
        from cosmos.thermo import carnot_efficiency
        for t_cold in [100, 200, 400]:
            eff = carnot_efficiency(500.0, t_cold)
            assert 0.0 < eff < 1.0

    def test_rms_speed_nitrogen(self):
        from cosmos.thermo import rms_speed
        # N2 at 300 K: ~517 m/s
        v = rms_speed(0.028, 300.0)
        assert abs(v - 517) < 20

    def test_stefan_boltzmann_flux(self):
        from cosmos.thermo import stefan_boltzmann_flux
        # Sun's surface ≈ 5778 K → ~63.2 MW/m²
        flux = stefan_boltzmann_flux(5778.0)
        assert abs(flux - 6.3e7) / 6.3e7 < 0.01

    def test_adiabatic_temperature(self):
        from cosmos.thermo import adiabatic_temperature
        T2 = adiabatic_temperature(300.0, 100000.0, 200000.0, gamma=1.4)
        assert T2 > 300.0  # compression heats


# ── nova.concurrent ──────────────────────────────────────────────────────────

class TestNovaConcurrent:
    def test_channel_send_recv(self):
        from nova.concurrent import channel
        ch = channel()
        ch.send(42)
        assert ch.recv() == 42

    def test_channel_fifo_order(self):
        from nova.concurrent import channel
        ch = channel()
        for i in range(5):
            ch.send(i)
        for i in range(5):
            assert ch.recv() == i

    def test_channel_bounded(self):
        from nova.concurrent import channel
        ch = channel(maxsize=2)
        ch.send("a")
        ch.send("b")
        # channel is full; try_recv should work
        assert ch.try_recv() == "a"

    def test_spawn_executes(self):
        from nova.concurrent import channel, spawn
        ch = channel()
        spawn(lambda: ch.send("done"))
        result = ch.recv(timeout_s=2.0)
        assert result == "done"

    def test_atomic_int_add(self):
        from nova.concurrent import atomic_int
        a = atomic_int(10)
        a.add(5)
        assert a.get() == 15

    def test_atomic_int_cas(self):
        from nova.concurrent import atomic_int
        a = atomic_int(0)
        ok = a.compare_and_swap(0, 100)
        assert ok and a.get() == 100
        fail = a.compare_and_swap(0, 999)
        assert not fail and a.get() == 100

    def test_parallel_map(self):
        from nova.concurrent import parallel_map
        result = parallel_map(lambda x: x * 2, [1, 2, 3, 4, 5])
        assert result == [2, 4, 6, 8, 10]


# ── nova.crypto ───────────────────────────────────────────────────────────────

class TestNovaCrypto:
    def test_sha256_known(self):
        from nova.crypto import sha256
        h = sha256(b"hello")
        assert h == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_sha256_string(self):
        from nova.crypto import sha256
        h = sha256("hello")
        assert len(h) == 64

    def test_sha256_deterministic(self):
        from nova.crypto import sha256
        assert sha256("nova") == sha256("nova")

    def test_hmac_sha256(self):
        from nova.crypto import hmac_sha256
        h = hmac_sha256("key", "message")
        assert len(h) == 64

    def test_random_hex_length(self):
        from nova.crypto import random_hex
        h = random_hex(16)
        assert len(h) == 32

    def test_random_hex_different(self):
        from nova.crypto import random_hex
        assert random_hex() != random_hex()

    def test_uuid4_format(self):
        from nova.crypto import uuid4
        u = uuid4()
        parts = u.split("-")
        assert len(parts) == 5

    def test_base64_roundtrip(self):
        from nova.crypto import base64_encode, base64_decode
        data = b"Hello, universe!"
        encoded = base64_encode(data)
        decoded = base64_decode(encoded)
        assert decoded == data

    def test_compare_digest_equal(self):
        from nova.crypto import compare_digest
        assert compare_digest("abc", "abc")

    def test_compare_digest_different(self):
        from nova.crypto import compare_digest
        assert not compare_digest("abc", "xyz")

    def test_pbkdf2(self):
        from nova.crypto import pbkdf2
        h = pbkdf2("password", "salt", iterations=1000)
        assert len(h) == 64
        assert h == pbkdf2("password", "salt", iterations=1000)


# ── nova.db ───────────────────────────────────────────────────────────────────

class TestNovaDB:
    def test_connect_in_memory(self):
        from nova.db import connect
        with connect(":memory:") as db:
            assert db is not None

    def test_create_table_and_insert(self):
        from nova.db import connect
        with connect(":memory:") as db:
            db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
            rowid = db.insert("t", {"val": "hello"})
            assert rowid == 1

    def test_query_all(self):
        from nova.db import connect
        with connect(":memory:") as db:
            db.execute("CREATE TABLE t (x INTEGER)")
            db.insert("t", {"x": 1})
            db.insert("t", {"x": 2})
            rows = db.query_all("SELECT * FROM t ORDER BY x")
            assert len(rows) == 2
            assert rows[0]["x"] == 1

    def test_query_one(self):
        from nova.db import connect
        with connect(":memory:") as db:
            db.execute("CREATE TABLE t (name TEXT)")
            db.insert("t", {"name": "nova"})
            row = db.query_one("SELECT * FROM t")
            assert row is not None
            assert row["name"] == "nova"

    def test_query_one_empty(self):
        from nova.db import connect
        with connect(":memory:") as db:
            db.execute("CREATE TABLE t (x INTEGER)")
            row = db.query_one("SELECT * FROM t")
            assert row is None

    def test_update(self):
        from nova.db import connect
        with connect(":memory:") as db:
            db.execute("CREATE TABLE t (id INTEGER, val TEXT)")
            db.insert("t", {"id": 1, "val": "old"})
            n = db.update("t", {"val": "new"}, "id = ?", (1,))
            assert n == 1
            row = db.query_one("SELECT val FROM t WHERE id = 1")
            assert row["val"] == "new"

    def test_delete(self):
        from nova.db import connect
        with connect(":memory:") as db:
            db.execute("CREATE TABLE t (x INTEGER)")
            db.insert("t", {"x": 42})
            n = db.delete("t", "x = ?", (42,))
            assert n == 1
            assert db.query_all("SELECT * FROM t") == []

    def test_table_exists(self):
        from nova.db import connect
        with connect(":memory:") as db:
            assert not db.table_exists("t")
            db.execute("CREATE TABLE t (x INTEGER)")
            assert db.table_exists("t")


# ── nova.test ─────────────────────────────────────────────────────────────────

class TestNovaTest:
    def test_assert_eq_passes(self):
        from nova.test import assert_eq
        assert_eq(1 + 1, 2)

    def test_assert_eq_fails(self):
        from nova.test import assert_eq
        with pytest.raises(AssertionError):
            assert_eq(1, 2)

    def test_assert_ne_passes(self):
        from nova.test import assert_ne
        assert_ne(1, 2)

    def test_assert_approx_passes(self):
        from nova.test import assert_approx
        assert_approx(3.14159, 3.14, tolerance=0.01)

    def test_assert_approx_fails(self):
        from nova.test import assert_approx
        with pytest.raises(AssertionError):
            assert_approx(3.0, 5.0, tolerance=0.1)

    def test_assert_raises_passes(self):
        from nova.test import assert_raises
        assert_raises(ValueError, lambda: int("not_a_number"))

    def test_assert_raises_fails_wrong_exception(self):
        from nova.test import assert_raises
        with pytest.raises(AssertionError):
            assert_raises(KeyError, lambda: int("not_a_number"))

    def test_assert_in(self):
        from nova.test import assert_in, assert_not_in
        assert_in(3, [1, 2, 3])
        assert_not_in(99, [1, 2, 3])

    def test_assert_true_false(self):
        from nova.test import assert_true, assert_false
        assert_true(1 == 1)
        assert_false(1 == 2)

    def test_assert_is_none(self):
        from nova.test import assert_is_none, assert_is_not_none
        assert_is_none(None)
        assert_is_not_none(42)

    def test_suite_runner(self):
        from nova.test import TestSuite
        results_box = []
        def my_test():
            results_box.append("ran")
        suite = TestSuite("demo")
        suite.add(my_test)
        results = suite.run()
        assert len(results) == 1
        assert results[0].passed
        assert results_box == ["ran"]

    def test_suite_captures_failures(self):
        from nova.test import TestSuite
        def failing():
            raise AssertionError("oops")
        suite = TestSuite("demo")
        suite.add(failing)
        results = suite.run()
        assert not results[0].passed
        assert "oops" in results[0].error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
