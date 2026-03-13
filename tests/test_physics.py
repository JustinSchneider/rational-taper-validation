"""Tests for the physics module: baryonic velocity, model fitters, and BIC."""

import pytest
import numpy as np
from scipy.special import i0, i1, k0, k1

from src.physics import (
    A0_MOND,
    G_PC,
    KPC_TO_PC,
    circular_velocity_thin_disk,
    compute_bic,
    compute_v_bary,
    fit_mond_fixed,
    fit_mond_free,
    fit_nfw,
    fit_rational_taper,
    mond_velocity,
    nfw_velocity,
)


# ---------------------------------------------------------------------------
# Baryonic velocity
# ---------------------------------------------------------------------------

class TestComputeVBary:
    def test_all_positive(self):
        v_gas = np.array([10.0])
        v_disk = np.array([20.0])
        v_bulge = np.array([0.0])
        result = compute_v_bary(v_gas, v_disk, v_bulge, upsilon_disk=0.5, upsilon_bulge=0.7)
        # V^2 = 10*10 + 0.5*20*20 + 0 = 100 + 200 = 300
        assert result[0] == pytest.approx(np.sqrt(300.0))

    def test_negative_vgas_reduces_vbary(self):
        v_gas_pos = np.array([10.0])
        v_gas_neg = np.array([-10.0])
        v_disk = np.array([20.0])
        v_bulge = np.array([0.0])

        v_pos = compute_v_bary(v_gas_pos, v_disk, v_bulge)
        v_neg = compute_v_bary(v_gas_neg, v_disk, v_bulge)
        assert v_neg[0] < v_pos[0]

    def test_output_always_nonnegative(self):
        v_gas = np.array([-50.0])
        v_disk = np.array([5.0])
        v_bulge = np.array([0.0])
        result = compute_v_bary(v_gas, v_disk, v_bulge)
        assert result[0] >= 0.0

    def test_upsilon_scaling(self):
        v_gas = np.array([0.0])
        v_disk = np.array([20.0])
        v_bulge = np.array([0.0])

        result_low = compute_v_bary(v_gas, v_disk, v_bulge, upsilon_disk=0.25)
        result_high = compute_v_bary(v_gas, v_disk, v_bulge, upsilon_disk=1.0)
        # Ratio should be sqrt(1.0/0.25) = 2.0
        assert (result_high[0] / result_low[0]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Thin-disk potential (Casertano 1983)
# ---------------------------------------------------------------------------

class TestCircularVelocityThinDisk:
    def test_point_mass_limit(self):
        """A narrow ring should give ~Keplerian V at R >> ring radius."""
        a = 1.0
        width = 0.05
        r_profile = np.linspace(a - 3 * width, a + 3 * width, 50)
        r_profile = r_profile[r_profile > 0]
        sigma_profile = 100.0 * np.exp(-0.5 * ((r_profile - a) / width) ** 2)

        r_eval = np.array([10.0, 20.0, 50.0])
        v = circular_velocity_thin_disk(r_eval, r_profile, sigma_profile)

        ratio_v = v[0] / v[1]
        ratio_expected = np.sqrt(r_eval[1] / r_eval[0])
        assert ratio_v == pytest.approx(ratio_expected, rel=0.05)

    def test_exponential_disk_freeman(self):
        """Compare against Freeman (1970) analytic result for an exponential disk."""
        R_d_kpc = 2.0
        Sigma_0 = 50.0

        r_profile = np.linspace(0.05, 12 * R_d_kpc, 500)
        sigma_profile = Sigma_0 * np.exp(-r_profile / R_d_kpc)

        r_eval = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
        v_numerical = circular_velocity_thin_disk(r_eval, r_profile, sigma_profile)

        R_d_pc = R_d_kpc * KPC_TO_PC
        y = (r_eval * KPC_TO_PC) / (2.0 * R_d_pc)
        v2_analytic = (
            4.0 * np.pi * G_PC * Sigma_0 * R_d_pc
            * y ** 2
            * (i0(y) * k0(y) - i1(y) * k1(y))
        )
        v_analytic = np.sqrt(np.maximum(v2_analytic, 0.0))

        for i in range(len(r_eval)):
            if v_analytic[i] > 1.0:
                assert v_numerical[i] == pytest.approx(v_analytic[i], rel=0.05)

    def test_returns_nonnegative(self):
        r_profile = np.array([1.0, 2.0, 3.0])
        sigma_profile = np.array([10.0, 5.0, 1.0])
        r_eval = np.linspace(0.1, 5.0, 20)
        v = circular_velocity_thin_disk(r_eval, r_profile, sigma_profile)
        assert np.all(v >= 0)

    def test_helium_factor_scaling(self):
        r_profile = np.linspace(0.5, 10.0, 30)
        sigma_profile = 5.0 * np.exp(-r_profile / 3.0)
        r_eval = np.array([2.0, 5.0, 8.0])

        v_bare = circular_velocity_thin_disk(r_eval, r_profile, sigma_profile, helium_factor=1.0)
        v_he = circular_velocity_thin_disk(r_eval, r_profile, sigma_profile, helium_factor=1.33)

        expected_ratio = np.sqrt(1.33)
        for i in range(len(r_eval)):
            if v_bare[i] > 1.0:
                assert (v_he[i] / v_bare[i]) == pytest.approx(expected_ratio, rel=0.01)


# ---------------------------------------------------------------------------
# BIC
# ---------------------------------------------------------------------------

class TestComputeBIC:
    def test_zero_params(self):
        bic = compute_bic(n_points=10, k_params=0, chi_squared=5.0)
        assert bic == pytest.approx(5.0)

    def test_penalizes_more_params(self):
        chi2 = 10.0
        n = 20
        bic_0 = compute_bic(n, 0, chi2)
        bic_1 = compute_bic(n, 1, chi2)
        bic_2 = compute_bic(n, 2, chi2)
        assert bic_0 < bic_1 < bic_2

    def test_lower_chi2_preferred(self):
        bic_good = compute_bic(10, 2, 5.0)
        bic_bad = compute_bic(10, 2, 50.0)
        assert bic_good < bic_bad


# ---------------------------------------------------------------------------
# NFW model
# ---------------------------------------------------------------------------

class TestNFWVelocity:
    def test_keplerian_limit(self):
        """At R >> R_200, V_NFW should decline."""
        radius = np.array([1.0, 5.0, 20.0, 100.0])
        v = nfw_velocity(radius, c=10.0, v_200=150.0)
        # Should peak somewhere and decline at very large R
        assert v[-1] < v[1]

    def test_always_nonnegative(self):
        radius = np.linspace(0.1, 100.0, 50)
        v = nfw_velocity(radius, c=10.0, v_200=150.0)
        assert np.all(v >= 0.0)

    def test_higher_concentration_more_concentrated(self):
        """Higher c should give relatively more power at small R."""
        radius = np.array([0.5, 1.0, 2.0])
        v_low_c = nfw_velocity(radius, c=5.0, v_200=150.0)
        v_high_c = nfw_velocity(radius, c=20.0, v_200=150.0)
        # At inner radii high-c halo should be relatively stronger
        assert v_high_c[0] > v_low_c[0]


class TestFitNFW:
    def test_converges_on_realistic_curve(self, synthetic_rotation_curve):
        """NFW should converge on a reasonable rotation curve."""
        data = synthetic_rotation_curve
        result = fit_nfw(
            data["radius"], data["v_obs"], data["v_err"], data["v_bary"],
            galaxy_id="SYNTHETIC",
        )
        assert result.converged
        assert result.model_name == "nfw"
        assert result.n_params == 2
        assert np.isfinite(result.bic)
        assert np.isfinite(result.param1)   # c
        assert np.isfinite(result.param2)   # V_200
        assert result.param1 > 0
        assert result.param2 > 0

    def test_rmse_nonnegative(self, synthetic_rotation_curve):
        data = synthetic_rotation_curve
        result = fit_nfw(
            data["radius"], data["v_obs"], data["v_err"], data["v_bary"],
        )
        if result.converged:
            assert result.residuals_rmse >= 0.0

    def test_to_dict_excludes_arrays(self, synthetic_rotation_curve):
        data = synthetic_rotation_curve
        result = fit_nfw(
            data["radius"], data["v_obs"], data["v_err"], data["v_bary"],
        )
        d = result.to_dict()
        for key, val in d.items():
            assert not isinstance(val, np.ndarray), f"Array found in to_dict(): {key}"


# ---------------------------------------------------------------------------
# MOND models
# ---------------------------------------------------------------------------

class TestMondVelocity:
    def test_newtonian_limit(self):
        """At high acceleration (large V_bary^2/R), V_MOND ≈ V_bary."""
        radius = np.array([0.1])   # very small R → high acceleration
        v_bary = np.array([200.0])  # very high V_bary → Newtonian regime
        v_mond = mond_velocity(radius, v_bary)
        # In Newtonian limit, correction is tiny
        assert v_mond[0] == pytest.approx(v_bary[0], rel=0.05)

    def test_deep_mond_limit(self):
        """At low acceleration, V_MOND > V_bary (extra boost expected)."""
        radius = np.array([50.0])   # large R → low baryonic acceleration
        v_bary = np.array([10.0])   # low V_bary
        v_mond = mond_velocity(radius, v_bary)
        assert v_mond[0] > v_bary[0]

    def test_zero_vbary_gives_zero(self):
        radius = np.array([1.0, 5.0, 10.0])
        v_bary = np.zeros(3)
        v_mond = mond_velocity(radius, v_bary)
        assert np.all(v_mond == 0.0)

    def test_always_nonnegative(self):
        radius = np.linspace(0.5, 30.0, 30)
        v_bary = 100.0 * np.sqrt(radius / (radius + 3.0))
        v_mond = mond_velocity(radius, v_bary)
        assert np.all(v_mond >= 0.0)

    def test_higher_a0_gives_higher_mond_velocity(self):
        """Higher a_0 should give a larger MOND boost at fixed V_bary, R."""
        radius = np.array([10.0])
        v_bary = np.array([50.0])
        v_low = mond_velocity(radius, v_bary, a0=1.0)
        v_high = mond_velocity(radius, v_bary, a0=10.0)
        assert v_high[0] > v_low[0]


class TestFitMondFixed:
    def test_no_free_params(self, synthetic_rotation_curve):
        data = synthetic_rotation_curve
        result = fit_mond_fixed(
            data["radius"], data["v_obs"], data["v_err"], data["v_bary"],
            galaxy_id="SYNTHETIC",
        )
        assert result.model_name == "mond_fixed"
        assert result.n_params == 0
        assert result.converged is True
        assert result.param1 == pytest.approx(A0_MOND)
        assert np.isfinite(result.chi_squared)
        assert np.isfinite(result.bic)

    def test_bic_equals_chi2_for_zero_params(self, synthetic_rotation_curve):
        """BIC = chi^2 + 0*ln(n) = chi^2 when k=0."""
        data = synthetic_rotation_curve
        result = fit_mond_fixed(
            data["radius"], data["v_obs"], data["v_err"], data["v_bary"],
        )
        assert result.bic == pytest.approx(result.chi_squared)


class TestFitMondFree:
    def test_converges(self, synthetic_rotation_curve):
        data = synthetic_rotation_curve
        result = fit_mond_free(
            data["radius"], data["v_obs"], data["v_err"], data["v_bary"],
            galaxy_id="SYNTHETIC",
        )
        assert result.model_name == "mond_free"
        assert result.n_params == 1
        assert result.converged
        assert np.isfinite(result.param1)
        # Best-fit a0 should be within the allowed bounds
        assert 1.0 <= result.param1 <= 10.0

    def test_free_bic_le_fixed_bic(self, synthetic_rotation_curve):
        """Free MOND (1 param) should have BIC <= Fixed MOND (0 params) when
        the best-fit a0 differs significantly from canonical."""
        data = synthetic_rotation_curve
        r_fixed = fit_mond_fixed(
            data["radius"], data["v_obs"], data["v_err"], data["v_bary"],
        )
        r_free = fit_mond_free(
            data["radius"], data["v_obs"], data["v_err"], data["v_bary"],
        )
        # Free MOND chi2 should be <= fixed MOND chi2 (more freedom)
        if r_free.converged:
            assert r_free.chi_squared <= r_fixed.chi_squared + 1e-6


# ---------------------------------------------------------------------------
# Rational Taper model
# ---------------------------------------------------------------------------

class TestFitRationalTaper:
    def test_recovers_known_parameters(self, synthetic_rotation_curve):
        """Fit synthetic data and verify omega and R_t recovered within tolerance."""
        data = synthetic_rotation_curve
        result = fit_rational_taper(
            data["radius"], data["v_obs"], data["v_err"], data["v_bary"],
            galaxy_id="SYNTHETIC",
        )
        assert result.converged
        assert result.model_name == "rational_taper"
        assert result.n_params == 2
        assert result.param1 == pytest.approx(data["true_omega"], abs=2.0)  # omega
        assert result.param2 == pytest.approx(data["true_rt"], abs=3.0)     # R_t

    def test_rmse_nonnegative(self, synthetic_rotation_curve):
        data = synthetic_rotation_curve
        result = fit_rational_taper(
            data["radius"], data["v_obs"], data["v_err"], data["v_bary"],
        )
        assert result.residuals_rmse >= 0.0

    def test_to_dict_excludes_arrays(self, synthetic_rotation_curve):
        data = synthetic_rotation_curve
        result = fit_rational_taper(
            data["radius"], data["v_obs"], data["v_err"], data["v_bary"],
        )
        d = result.to_dict()
        for key, val in d.items():
            assert not isinstance(val, np.ndarray), f"Array found in to_dict(): {key}"

    def test_v_sat_derivable(self, synthetic_rotation_curve):
        """V_sat = omega * R_t should be finite and positive."""
        data = synthetic_rotation_curve
        result = fit_rational_taper(
            data["radius"], data["v_obs"], data["v_err"], data["v_bary"],
        )
        if result.converged:
            v_sat = result.param1 * result.param2
            assert np.isfinite(v_sat)
            assert v_sat > 0

    def test_zero_errors_handled(self):
        """Zero errors should not cause division by zero."""
        radius = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v_bary = np.array([50.0, 60.0, 65.0, 68.0, 70.0])
        v_obs = v_bary + np.array([5.0, 8.0, 10.0, 9.0, 8.0])
        v_err = np.array([0.0, 3.0, 3.0, 3.0, 3.0])

        result = fit_rational_taper(radius, v_obs, v_err, v_bary)
        assert result.converged
        assert np.isfinite(result.param1)


# ---------------------------------------------------------------------------
# Cross-model BIC comparisons (sanity checks)
# ---------------------------------------------------------------------------

class TestBICComparisons:
    def test_all_models_produce_bic(self, synthetic_rotation_curve):
        """All four models should produce a finite BIC value."""
        data = synthetic_rotation_curve
        kwargs = dict(
            radius=data["radius"],
            v_obs=data["v_obs"],
            v_err=data["v_err"],
            v_bary=data["v_bary"],
        )
        results = [
            fit_nfw(**kwargs),
            fit_mond_fixed(**kwargs),
            fit_mond_free(**kwargs),
            fit_rational_taper(**kwargs),
        ]
        for r in results:
            if r.converged:
                assert np.isfinite(r.bic), f"{r.model_name} produced non-finite BIC"

    def test_mond_fixed_bic_penalizes_least(self, synthetic_rotation_curve):
        """Fixed MOND has 0 params so its BIC penalty term is zero."""
        data = synthetic_rotation_curve
        kwargs = dict(
            radius=data["radius"],
            v_obs=data["v_obs"],
            v_err=data["v_err"],
            v_bary=data["v_bary"],
        )
        r_mond = fit_mond_fixed(**kwargs)
        r_nfw = fit_nfw(**kwargs)
        r_taper = fit_rational_taper(**kwargs)

        # At equal chi2, MOND fixed always wins on BIC (0 params)
        # Here we just verify the BIC formula is correctly applied
        if r_mond.converged and r_nfw.converged:
            n = r_mond.n_points
            bic_mond_manual = r_mond.chi_squared + 0 * np.log(n)
            assert r_mond.bic == pytest.approx(bic_mond_manual)

        if r_taper.converged:
            n = r_taper.n_points
            bic_taper_manual = r_taper.chi_squared + 2 * np.log(n)
            assert r_taper.bic == pytest.approx(bic_taper_manual)
