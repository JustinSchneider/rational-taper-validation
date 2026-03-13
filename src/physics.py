"""Core physics equations for rotation curve decomposition and model fitting.

Three models are compared against the same baryonic mass decomposition:
  - NFW Halo (ΛCDM): 2 parameters (c, V_200)
  - MOND (Simple interpolation): 0 free parameters (fixed a_0) or 1 (free a_0)
  - Rational Taper (Schneider 2026): 2 parameters (omega, R_t)

All models use the same baryonic velocity convention from Lelli et al. (2016):
  V_bary = sqrt(|V_gas|*V_gas + Upsilon_d*|V_disk|*V_disk + Upsilon_b*|V_bulge|*V_bulge)

Key references:
  - Lelli, McGaugh, & Schombert (2016) AJ 152, 157  — SPARC data and Eq. 2
  - Navarro, Frenk, & White (1996) ApJ 462, 563     — NFW profile
  - Milgrom (1983) ApJ 270, 365                     — MOND
  - Famaey & McGaugh (2012) LRR 15, 10              — MOND interpolation functions
  - Casertano (1983) MNRAS 203, 735                  — Thin-disk potential
  - Binney & Tremaine (2008) Eq. 2.188              — Ring potential
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import ellipk

from src.utils import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Gravitational constant in units: pc * (km/s)^2 / M_sun
G_PC = 4.302e-3

# Multiplicative factor to convert HI mass to total gas mass (HI + He)
HELIUM_FACTOR = 1.33

# 1 kpc = 1000 pc
KPC_TO_PC = 1000.0

# MOND acceleration scale: a_0 = 1.2e-10 m/s^2, converted to km^2/s^2/kpc
# 1 kpc = 3.0857e19 m;  1 km^2 = 1e6 m^2
# a_0 [km^2/s^2/kpc] = 1.2e-10 [m/s^2] * 3.0857e19 [m/kpc] / 1e6 [m^2/km^2]
#                     = 1.2e-10 * 3.0857e13 = 3703 km^2 s^-2 kpc^-1
# Check: g = a_0 * R -> (km^2/s^2/kpc) * kpc = km^2/s^2 ✓
A0_MOND = 3703.0  # km^2 s^-2 kpc^-1  (= 1.2e-10 m/s^2 in these units)

# Hubble constant for NFW R_200 calculation
H0_KM_S_MPC = 73.0  # km/s/Mpc
H0_KM_S_KPC = H0_KM_S_MPC / 1000.0  # km/s/kpc


# ---------------------------------------------------------------------------
# Thin-disk gravitational potential (Casertano 1983)
# ---------------------------------------------------------------------------


def circular_velocity_thin_disk(
    r_eval: np.ndarray,
    r_profile: np.ndarray,
    sigma_profile: np.ndarray,
    helium_factor: float = 1.0,
) -> np.ndarray:
    """Compute circular velocity from a tabulated surface density profile.

    Uses the Casertano (1983) method: the disk surface density is interpolated
    onto a fine, evenly-spaced grid and the gravitational potential is summed
    from ring contributions using complete elliptic integrals. The circular
    velocity is obtained by numerical differentiation of the potential.

    A small softening length (equivalent to finite disk thickness) avoids the
    logarithmic singularity that occurs when evaluation and ring radii coincide.

    Based on Binney & Tremaine (2008), Eq. 2.188 for the ring potential:
        Phi(R) = -G * dM / (pi * (R + a)) * K(k^2)
    where k^2 = 4*R*a / (R + a)^2.

    Args:
        r_eval: Radii at which to evaluate V_circ (kpc).
        r_profile: Radii of the surface density samples (kpc).
        sigma_profile: Surface density at each r_profile (M_sun/pc^2).
        helium_factor: Multiplicative factor for sigma (use 1.33 for HI gas
            to account for helium; use 1.0 for stellar mass).

    Returns:
        Circular velocity (km/s) at each r_eval. Always non-negative.
    """
    r_eval = np.asarray(r_eval, dtype=np.float64)
    r_profile = np.asarray(r_profile, dtype=np.float64)
    sigma_profile = np.asarray(sigma_profile, dtype=np.float64)

    R_eval_pc = r_eval * KPC_TO_PC
    Rp = r_profile * KPC_TO_PC
    sigma = sigma_profile * helium_factor

    r_min_pc = max(Rp.min() * 0.5, 5.0)
    r_max_pc = max(Rp.max(), R_eval_pc.max()) * 1.3
    n_fine = max(5000, len(Rp) * 20)
    R_fine = np.linspace(r_min_pc, r_max_pc, n_fine)
    sigma_fine = np.interp(R_fine, Rp, sigma, left=0.0, right=0.0)

    dr_fine = R_fine[1] - R_fine[0]
    dM_fine = 2.0 * np.pi * R_fine * sigma_fine * dr_fine

    eps = dr_fine * 0.1
    eps2 = eps ** 2

    R_grid = R_fine + dr_fine * 0.5
    Phi_grid = np.zeros_like(R_grid)
    for j in range(n_fine):
        if dM_fine[j] <= 0:
            continue
        sum_R_sq = (R_grid + R_fine[j]) ** 2 + eps2
        k2 = 4.0 * R_grid * R_fine[j] / sum_R_sq
        k2 = np.clip(k2, 0.0, 1.0 - 1e-10)
        Phi_grid += -2.0 * G_PC * dM_fine[j] / (np.pi * np.sqrt(sum_R_sq)) * ellipk(k2)

    dPhi_dR = np.gradient(Phi_grid, R_grid)
    v2_grid = R_grid * dPhi_dR
    v2_grid = np.maximum(v2_grid, 0.0)
    v_grid = np.sqrt(v2_grid)

    v_circ = np.interp(R_eval_pc, R_grid, v_grid, left=0.0, right=0.0)
    return v_circ


# ---------------------------------------------------------------------------
# Baryonic velocity computation
# ---------------------------------------------------------------------------


def compute_v_bary(
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    upsilon_disk: float = 0.5,
    upsilon_bulge: float = 0.7,
    gas_scale: float = 1.0,
) -> np.ndarray:
    """Compute total baryonic velocity using the Lelli et al. (2016) Eq. 2 convention.

    V_bary(R) = sqrt(|V_gas|*V_gas + Upsilon_d * |V_disk|*V_disk
                     + Upsilon_b * |V_bulge|*V_bulge)

    The |V|*V pattern preserves the sign: if V is negative (e.g., gas central
    depression), its squared contribution is negative, reducing V_bary.

    SPARC provides V_disk and V_bulge at Upsilon=1, so Upsilon enters as a
    direct multiplier on V^2 (equivalent to sqrt(Upsilon) scaling on V).

    Args:
        v_gas: Gas velocity component (km/s). May contain negative values.
        v_disk: Disk velocity component at Upsilon=1 (km/s).
        v_bulge: Bulge velocity component at Upsilon=1 (km/s).
        upsilon_disk: Stellar mass-to-light ratio for disk. Default 0.5.
        upsilon_bulge: Stellar mass-to-light ratio for bulge. Default 0.7.
        gas_scale: Multiplicative scaling on gas contribution. Default 1.0.

    Returns:
        Total baryonic velocity (km/s). Always non-negative.
    """
    v_gas = np.asarray(v_gas, dtype=np.float64)
    v_disk = np.asarray(v_disk, dtype=np.float64)
    v_bulge = np.asarray(v_bulge, dtype=np.float64)

    v2_gas = gas_scale * np.abs(v_gas) * v_gas
    v2_disk = upsilon_disk * np.abs(v_disk) * v_disk
    v2_bulge = upsilon_bulge * np.abs(v_bulge) * v_bulge

    v2_total = v2_gas + v2_disk + v2_bulge
    return np.sqrt(np.maximum(0.0, v2_total))


# ---------------------------------------------------------------------------
# BIC
# ---------------------------------------------------------------------------


def compute_bic(n_points: int, k_params: int, chi_squared: float) -> float:
    """Compute the Bayesian Information Criterion (BIC).

    BIC = chi^2 + k * ln(n)

    Lower BIC indicates a preferred model. The difference Delta_BIC follows
    the Kass & Raftery (1995) scale:
      |Delta_BIC| < 2:    Not worth mentioning
      2–6:                Positive evidence
      6–10:               Strong evidence
      > 10:               Very strong evidence

    Args:
        n_points: Number of data points.
        k_params: Number of free parameters in the model.
        chi_squared: Total (non-reduced) chi-squared value.

    Returns:
        BIC value.
    """
    return chi_squared + k_params * np.log(n_points)


# ---------------------------------------------------------------------------
# Shared error-handling helper
# ---------------------------------------------------------------------------


def _safe_errors(v_err: np.ndarray, galaxy_id: str = "unknown") -> np.ndarray:
    """Replace zero/negative errors with the minimum nonzero error (floor 1.0 km/s)."""
    positive_errs = v_err[v_err > 0]
    min_err = float(np.min(positive_errs)) if len(positive_errs) > 0 else 1.0
    v_err_safe = np.where(v_err > 0, v_err, min_err)
    if np.any(v_err <= 0):
        logger.warning(
            "%s: replaced %d zero/negative errors with %.2f km/s",
            galaxy_id, np.sum(v_err <= 0), min_err,
        )
    return v_err_safe


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelFitResult:
    """Container for results from any single-model fit."""

    galaxy_id: str
    model_name: str        # "nfw", "mond_fixed", "mond_free", "rational_taper"
    n_params: int          # free parameters (for BIC)
    chi_squared: float
    reduced_chi_squared: float
    bic: float
    residuals_rmse: float
    n_points: int
    converged: bool
    flag_v_obs_lt_v_bary: bool
    method_version: str
    upsilon_disk: float
    upsilon_bulge: float
    # Arrays for plotting (not stored in DB)
    v_bary: np.ndarray = field(repr=False)
    v_model: np.ndarray = field(repr=False)
    residuals: np.ndarray = field(repr=False)
    # Model-specific scalar parameters (stored as optional floats)
    param1: Optional[float] = None   # omega (taper/linear), c (NFW), a0 (MOND free)
    param1_err: Optional[float] = None
    param2: Optional[float] = None   # R_t (taper), V_200 (NFW)
    param2_err: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dict suitable for database insertion (excludes arrays)."""
        return {
            "galaxy_id": self.galaxy_id,
            "model_name": self.model_name,
            "n_params": self.n_params,
            "chi_squared": self.chi_squared,
            "reduced_chi_squared": self.reduced_chi_squared,
            "bic": self.bic,
            "residuals_rmse": self.residuals_rmse,
            "n_points": self.n_points,
            "converged": self.converged,
            "flag_v_obs_lt_v_bary": self.flag_v_obs_lt_v_bary,
            "method_version": self.method_version,
            "upsilon_disk": self.upsilon_disk,
            "upsilon_bulge": self.upsilon_bulge,
            "param1": self.param1,
            "param1_err": self.param1_err,
            "param2": self.param2,
            "param2_err": self.param2_err,
        }


# ---------------------------------------------------------------------------
# Model A: NFW Halo (ΛCDM)
# ---------------------------------------------------------------------------


def nfw_velocity(
    radius: np.ndarray,
    c: float,
    v_200: float,
    h0: float = H0_KM_S_KPC,
) -> np.ndarray:
    """Compute the NFW halo circular velocity.

    Model:
        V_NFW^2(R) = V_200^2 * [ln(1 + c*x) - c*x/(1 + c*x)]
                              / {x * [ln(1 + c) - c/(1 + c)]}
    where x = R / R_200  and  R_200 = V_200 / (10 * H_0).

    Args:
        radius: Radial positions (kpc).
        c: Concentration parameter (dimensionless). Must be > 0.
        v_200: Virial velocity (km/s), defined at R_200. Must be > 0.
        h0: Hubble constant in km/s/kpc. Default 73 km/s/Mpc converted.

    Returns:
        NFW circular velocity (km/s) at each radius. Always non-negative.
    """
    r_200 = v_200 / (10.0 * h0)  # kpc
    x = radius / r_200

    # Numerator: ln(1 + c*x) - c*x / (1 + c*x)
    cx = c * x
    numerator = np.log1p(cx) - cx / (1.0 + cx)

    # Denominator factor: ln(1 + c) - c / (1 + c)
    denominator_factor = np.log1p(c) - c / (1.0 + c)

    v2 = v_200**2 * numerator / (x * denominator_factor)
    return np.sqrt(np.maximum(v2, 0.0))


def fit_nfw(
    radius: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    v_bary: np.ndarray,
    galaxy_id: str = "unknown",
    method_version: str = "v1_nfw",
    upsilon_disk: float = 0.5,
    upsilon_bulge: float = 0.7,
    c_bounds: tuple = (1.0, 100.0),
    v200_bounds: tuple = (1.0, 2000.0),
) -> ModelFitResult:
    """Fit the NFW halo model to a rotation curve.

    Total model: V_total = sqrt(V_bary^2 + V_NFW^2)
    (quadrature, since NFW is an independent gravitational potential)

    Parameters: c (concentration), V_200 (virial velocity, km/s).

    Args:
        radius: Radial positions (kpc).
        v_obs: Observed velocities (km/s).
        v_err: Velocity errors (km/s). Values <= 0 are replaced.
        v_bary: Pre-computed baryonic velocity (km/s).
        galaxy_id: Galaxy identifier.
        method_version: Version string for reproducibility.
        upsilon_disk: Mass-to-light ratio used for V_bary.
        upsilon_bulge: Mass-to-light ratio used for V_bary.
        c_bounds: (lower, upper) bounds for concentration.
        v200_bounds: (lower, upper) bounds for V_200 (km/s).

    Returns:
        ModelFitResult with param1=c, param2=V_200.
    """
    radius = np.asarray(radius, dtype=np.float64)
    v_obs = np.asarray(v_obs, dtype=np.float64)
    v_err = np.asarray(v_err, dtype=np.float64)
    v_bary = np.asarray(v_bary, dtype=np.float64)

    n_points = len(radius)
    v_err_safe = _safe_errors(v_err, galaxy_id)
    flag_v_obs_lt_v_bary = bool(np.any(v_obs < v_bary))

    def _model(r, c, v_200):
        v_nfw = nfw_velocity(r, c, v_200)
        return np.sqrt(v_bary**2 + v_nfw**2)

    # Multi-start: vary concentration and virial velocity initial guesses
    initial_guesses = [
        (10.0, 150.0),
        (5.0, 100.0),
        (20.0, 200.0),
        (15.0, 50.0),
    ]

    best_chi2 = np.inf
    best_popt = None
    best_pcov = None
    converged = False

    for p0 in initial_guesses:
        try:
            popt, pcov = curve_fit(
                _model,
                radius,
                v_obs,
                p0=list(p0),
                sigma=v_err_safe,
                absolute_sigma=True,
                bounds=([c_bounds[0], v200_bounds[0]], [c_bounds[1], v200_bounds[1]]),
                maxfev=5000,
            )
            v_model_trial = _model(radius, *popt)
            res = v_obs - v_model_trial
            chi2_trial = float(np.sum((res / v_err_safe) ** 2))
            if chi2_trial < best_chi2:
                best_chi2 = chi2_trial
                best_popt = popt
                best_pcov = pcov
                converged = True
        except RuntimeError:
            pass

    if not converged:
        logger.warning("%s [NFW]: fit did not converge for any initial guess", galaxy_id)
        c_best = v200_best = c_err = v200_err = float("nan")
        v_model = np.full_like(radius, float("nan"))
        residuals = np.full_like(radius, float("nan"))
        chi2 = float("nan")
        reduced_chi2 = float("nan")
        rmse = float("nan")
    else:
        c_best = float(best_popt[0])
        v200_best = float(best_popt[1])
        perr = np.sqrt(np.diag(best_pcov))
        c_err = float(perr[0])
        v200_err = float(perr[1])
        v_model = _model(radius, c_best, v200_best)
        residuals = v_obs - v_model
        chi2 = best_chi2
        dof = max(n_points - 2, 1)
        reduced_chi2 = chi2 / dof
        rmse = float(np.sqrt(np.mean(residuals**2)))
        logger.info(
            "%s [NFW]: c=%.3f +/- %.3f  V_200=%.1f +/- %.1f  chi2_r=%.2f  RMSE=%.2f",
            galaxy_id, c_best, c_err, v200_best, v200_err, reduced_chi2, rmse,
        )

    bic = compute_bic(n_points, 2, chi2) if np.isfinite(chi2) else float("nan")

    return ModelFitResult(
        galaxy_id=galaxy_id,
        model_name="nfw",
        n_params=2,
        chi_squared=chi2,
        reduced_chi_squared=reduced_chi2,
        bic=bic,
        residuals_rmse=rmse,
        n_points=n_points,
        converged=converged,
        flag_v_obs_lt_v_bary=flag_v_obs_lt_v_bary,
        method_version=method_version,
        upsilon_disk=upsilon_disk,
        upsilon_bulge=upsilon_bulge,
        v_bary=v_bary,
        v_model=v_model if converged else np.full_like(radius, float("nan")),
        residuals=residuals if converged else np.full_like(radius, float("nan")),
        param1=c_best if converged else float("nan"),
        param1_err=c_err if converged else float("nan"),
        param2=v200_best if converged else float("nan"),
        param2_err=v200_err if converged else float("nan"),
    )


# ---------------------------------------------------------------------------
# Model B: MOND (Simple interpolation function)
# ---------------------------------------------------------------------------


def mond_velocity(
    radius: np.ndarray,
    v_bary: np.ndarray,
    a0: float = A0_MOND,
) -> np.ndarray:
    """Compute the MOND predicted velocity using the 'Simple' interpolation function.

    Famaey & Binney (2005) "Simple" function:
        mu(x) = x / (1 + x),  x = g / a_0

    The resulting total velocity satisfies:
        V_MOND^2 = V_bary^2 + (V_bary^2 / 2) * (sqrt(1 + 4*a_0*R/V_bary^2) - 1)

    This is the closed-form solution for the Simple mu function with
    circular velocity boundary conditions.

    At low acceleration (V_bary^2/R << a_0): V^4 → V_bary^2 * a_0 * R  (deep MOND)
    At high acceleration (V_bary^2/R >> a_0): V → V_bary  (Newtonian limit)

    Note: Where V_bary = 0, V_MOND = 0 (no baryons → no gravity).

    Args:
        radius: Radial positions (kpc).
        v_bary: Baryonic circular velocity (km/s). May not be zero everywhere.
        a0: MOND acceleration scale (km^2/s^2/kpc). Default A0_MOND.

    Returns:
        MOND total velocity (km/s). Always non-negative.
    """
    radius = np.asarray(radius, dtype=np.float64)
    v_bary = np.asarray(v_bary, dtype=np.float64)

    v2_bary = v_bary ** 2

    # Avoid division by zero where V_bary = 0
    safe_mask = v2_bary > 0.0

    v_mond = np.zeros_like(radius)
    r = radius[safe_mask]
    v2 = v2_bary[safe_mask]

    # Simple interpolation: V^2 = V_bary^2 + (V_bary^2/2)*(sqrt(1 + 4*a0*R/V_bary^2) - 1)
    inner = np.sqrt(1.0 + 4.0 * a0 * r / v2)
    v2_mond = v2 + (v2 / 2.0) * (inner - 1.0)
    v_mond[safe_mask] = np.sqrt(np.maximum(v2_mond, 0.0))

    return v_mond


def fit_mond_fixed(
    radius: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    v_bary: np.ndarray,
    galaxy_id: str = "unknown",
    method_version: str = "v1_mond_fixed",
    upsilon_disk: float = 0.5,
    upsilon_bulge: float = 0.7,
    a0: float = A0_MOND,
) -> ModelFitResult:
    """Evaluate Fixed MOND (zero free parameters) against a rotation curve.

    Since there are no free parameters, this is a pure prediction — no
    optimization is performed. Chi-squared is computed directly from the
    MOND prediction with canonical a_0.

    Args:
        radius: Radial positions (kpc).
        v_obs: Observed velocities (km/s).
        v_err: Velocity errors (km/s). Values <= 0 are replaced.
        v_bary: Pre-computed baryonic velocity (km/s).
        galaxy_id: Galaxy identifier.
        method_version: Version string for reproducibility.
        upsilon_disk: Mass-to-light ratio used for V_bary.
        upsilon_bulge: Mass-to-light ratio used for V_bary.
        a0: MOND acceleration scale to use. Default is canonical A0_MOND.

    Returns:
        ModelFitResult with 0 free parameters and param1=a0 (fixed, not fit).
    """
    radius = np.asarray(radius, dtype=np.float64)
    v_obs = np.asarray(v_obs, dtype=np.float64)
    v_err = np.asarray(v_err, dtype=np.float64)
    v_bary = np.asarray(v_bary, dtype=np.float64)

    n_points = len(radius)
    v_err_safe = _safe_errors(v_err, galaxy_id)
    flag_v_obs_lt_v_bary = bool(np.any(v_obs < v_bary))

    v_model = mond_velocity(radius, v_bary, a0=a0)
    residuals = v_obs - v_model
    chi2 = float(np.sum((residuals / v_err_safe) ** 2))
    dof = max(n_points, 1)  # 0 free parameters
    reduced_chi2 = chi2 / dof
    rmse = float(np.sqrt(np.mean(residuals**2)))
    bic = compute_bic(n_points, 0, chi2)

    logger.info(
        "%s [MOND fixed]: a0=%.4f  chi2_r=%.2f  RMSE=%.2f km/s",
        galaxy_id, a0, reduced_chi2, rmse,
    )

    return ModelFitResult(
        galaxy_id=galaxy_id,
        model_name="mond_fixed",
        n_params=0,
        chi_squared=chi2,
        reduced_chi_squared=reduced_chi2,
        bic=bic,
        residuals_rmse=rmse,
        n_points=n_points,
        converged=True,
        flag_v_obs_lt_v_bary=flag_v_obs_lt_v_bary,
        method_version=method_version,
        upsilon_disk=upsilon_disk,
        upsilon_bulge=upsilon_bulge,
        v_bary=v_bary,
        v_model=v_model,
        residuals=residuals,
        param1=a0,
        param1_err=0.0,
    )


def fit_mond_free(
    radius: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    v_bary: np.ndarray,
    galaxy_id: str = "unknown",
    method_version: str = "v1_mond_free",
    upsilon_disk: float = 0.5,
    upsilon_bulge: float = 0.7,
    a0_bounds: tuple = (1000.0, 10000.0),
) -> ModelFitResult:
    """Fit MOND with a_0 as a free parameter (1-parameter model).

    Allowing a_0 to vary by ~factor 3 around the canonical value acts as a
    proxy for distance uncertainty and systematic baryonic modeling errors.

    Args:
        radius: Radial positions (kpc).
        v_obs: Observed velocities (km/s).
        v_err: Velocity errors (km/s). Values <= 0 are replaced.
        v_bary: Pre-computed baryonic velocity (km/s).
        galaxy_id: Galaxy identifier.
        method_version: Version string for reproducibility.
        upsilon_disk: Mass-to-light ratio used for V_bary.
        upsilon_bulge: Mass-to-light ratio used for V_bary.
        a0_bounds: (lower, upper) bounds for a_0 in km^2/s^2/kpc.
                   Default (1000.0, 10000.0) spans ~0.27–2.7 times canonical value (3703).

    Returns:
        ModelFitResult with 1 free parameter and param1=best-fit a_0.
    """
    radius = np.asarray(radius, dtype=np.float64)
    v_obs = np.asarray(v_obs, dtype=np.float64)
    v_err = np.asarray(v_err, dtype=np.float64)
    v_bary = np.asarray(v_bary, dtype=np.float64)

    n_points = len(radius)
    v_err_safe = _safe_errors(v_err, galaxy_id)
    flag_v_obs_lt_v_bary = bool(np.any(v_obs < v_bary))

    def _model(r, a0):
        return mond_velocity(r, v_bary, a0=a0)

    try:
        popt, pcov = curve_fit(
            _model,
            radius,
            v_obs,
            p0=[A0_MOND],
            sigma=v_err_safe,
            absolute_sigma=True,
            bounds=([a0_bounds[0]], [a0_bounds[1]]),
        )
        a0_best = float(popt[0])
        a0_err = float(np.sqrt(np.diag(pcov))[0])
        converged = True
    except RuntimeError as e:
        logger.warning("%s [MOND free]: fit did not converge — %s", galaxy_id, e)
        a0_best = a0_err = float("nan")
        converged = False

    if converged:
        v_model = _model(radius, a0_best)
    else:
        v_model = np.full_like(radius, float("nan"))

    residuals = v_obs - v_model
    chi2 = float(np.sum((residuals / v_err_safe) ** 2)) if converged else float("nan")
    dof = max(n_points - 1, 1)
    reduced_chi2 = chi2 / dof if converged else float("nan")
    rmse = float(np.sqrt(np.mean(residuals**2))) if converged else float("nan")
    bic = compute_bic(n_points, 1, chi2) if converged else float("nan")

    if converged:
        logger.info(
            "%s [MOND free]: a0=%.4f +/- %.4f  chi2_r=%.2f  RMSE=%.2f km/s",
            galaxy_id, a0_best, a0_err, reduced_chi2, rmse,
        )

    return ModelFitResult(
        galaxy_id=galaxy_id,
        model_name="mond_free",
        n_params=1,
        chi_squared=chi2,
        reduced_chi_squared=reduced_chi2,
        bic=bic,
        residuals_rmse=rmse,
        n_points=n_points,
        converged=converged,
        flag_v_obs_lt_v_bary=flag_v_obs_lt_v_bary,
        method_version=method_version,
        upsilon_disk=upsilon_disk,
        upsilon_bulge=upsilon_bulge,
        v_bary=v_bary,
        v_model=v_model,
        residuals=residuals,
        param1=a0_best,
        param1_err=a0_err,
    )


# ---------------------------------------------------------------------------
# Model C: Rational Taper (Schneider 2026)
# ---------------------------------------------------------------------------


def fit_rational_taper(
    radius: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    v_bary: np.ndarray,
    galaxy_id: str = "unknown",
    method_version: str = "v1_rational_taper",
    upsilon_disk: float = 0.5,
    upsilon_bulge: float = 0.7,
    omega_bounds: tuple = (0.0, 200.0),
    rt_bounds: tuple = (0.1, None),
) -> ModelFitResult:
    """Fit the Rational Taper model to a rotation curve.

    Model: V_model = V_bary + omega * R / (1 + R / R_t)

    At small R: correction → omega * R  (linear)
    At large R: correction → omega * R_t = V_sat  (constant)

    Additive (not quadrature) coupling, consistent with the previous
    baryonic-omega-analysis project and the FC25 baseline.

    Multi-start optimizer with four initial conditions; retains the
    lowest chi-squared solution to avoid local minima.

    Args:
        radius: Radial positions (kpc).
        v_obs: Observed velocities (km/s).
        v_err: Velocity errors (km/s). Values <= 0 are replaced.
        v_bary: Pre-computed baryonic velocity (km/s).
        galaxy_id: Galaxy identifier.
        method_version: Version string for reproducibility.
        upsilon_disk: Mass-to-light ratio used for V_bary.
        upsilon_bulge: Mass-to-light ratio used for V_bary.
        omega_bounds: (lower, upper) bounds for omega (km/s/kpc).
        rt_bounds: (lower, upper) bounds for R_t (kpc).
                   If upper is None, defaults to 5 * max(radius).

    Returns:
        ModelFitResult with param1=omega, param2=R_t.
        Derived V_sat = omega * R_t accessible via param1 * param2.
    """
    radius = np.asarray(radius, dtype=np.float64)
    v_obs = np.asarray(v_obs, dtype=np.float64)
    v_err = np.asarray(v_err, dtype=np.float64)
    v_bary = np.asarray(v_bary, dtype=np.float64)

    n_points = len(radius)
    v_err_safe = _safe_errors(v_err, galaxy_id)
    flag_v_obs_lt_v_bary = bool(np.any(v_obs < v_bary))

    r_max = float(np.max(radius))
    rt_upper = rt_bounds[1] if rt_bounds[1] is not None else 5.0 * r_max

    def _model(r, omega, r_t):
        return v_bary + omega * r / (1.0 + r / r_t)

    # Four initial conditions (same as Phase III in sibling project)
    initial_guesses = [
        (5.0, 5.0),
        (10.0, 2.0),
        (2.0, 15.0),
        (20.0, r_max),
    ]

    best_chi2 = np.inf
    best_popt = None
    best_pcov = None
    converged = False

    for p0 in initial_guesses:
        try:
            popt, pcov = curve_fit(
                _model,
                radius,
                v_obs,
                p0=list(p0),
                sigma=v_err_safe,
                absolute_sigma=True,
                bounds=([omega_bounds[0], rt_bounds[0]], [omega_bounds[1], rt_upper]),
                maxfev=5000,
            )
            v_trial = _model(radius, *popt)
            res = v_obs - v_trial
            chi2_trial = float(np.sum((res / v_err_safe) ** 2))
            if chi2_trial < best_chi2:
                best_chi2 = chi2_trial
                best_popt = popt
                best_pcov = pcov
                converged = True
        except RuntimeError:
            pass

    if not converged:
        logger.warning(
            "%s [Rational Taper]: fit did not converge for any initial guess", galaxy_id
        )
        omega_best = rt_best = omega_err = rt_err = float("nan")
        v_model = np.full_like(radius, float("nan"))
        residuals = np.full_like(radius, float("nan"))
        chi2 = float("nan")
        reduced_chi2 = float("nan")
        rmse = float("nan")
    else:
        omega_best = float(best_popt[0])
        rt_best = float(best_popt[1])
        perr = np.sqrt(np.diag(best_pcov))
        omega_err = float(perr[0])
        rt_err = float(perr[1])
        v_model = _model(radius, omega_best, rt_best)
        residuals = v_obs - v_model
        chi2 = best_chi2
        dof = max(n_points - 2, 1)
        reduced_chi2 = chi2 / dof
        rmse = float(np.sqrt(np.mean(residuals**2)))
        v_sat = omega_best * rt_best
        logger.info(
            "%s [Rational Taper]: omega=%.4f +/- %.4f  R_t=%.4f +/- %.4f"
            "  V_sat=%.2f  chi2_r=%.2f  RMSE=%.2f",
            galaxy_id, omega_best, omega_err, rt_best, rt_err,
            v_sat, reduced_chi2, rmse,
        )

    bic = compute_bic(n_points, 2, chi2) if np.isfinite(chi2) else float("nan")

    return ModelFitResult(
        galaxy_id=galaxy_id,
        model_name="rational_taper",
        n_params=2,
        chi_squared=chi2,
        reduced_chi_squared=reduced_chi2,
        bic=bic,
        residuals_rmse=rmse,
        n_points=n_points,
        converged=converged,
        flag_v_obs_lt_v_bary=flag_v_obs_lt_v_bary,
        method_version=method_version,
        upsilon_disk=upsilon_disk,
        upsilon_bulge=upsilon_bulge,
        v_bary=v_bary,
        v_model=v_model if converged else np.full_like(radius, float("nan")),
        residuals=residuals if converged else np.full_like(radius, float("nan")),
        param1=omega_best if converged else float("nan"),
        param1_err=omega_err if converged else float("nan"),
        param2=rt_best if converged else float("nan"),
        param2_err=rt_err if converged else float("nan"),
    )


# ---------------------------------------------------------------------------
# Acceleration diagnostics (for Phase III physical analysis)
# ---------------------------------------------------------------------------


def compute_total_acceleration(
    radius: np.ndarray,
    v_total: np.ndarray,
) -> np.ndarray:
    """Compute centripetal acceleration g = V^2 / R.

    Args:
        radius: Radial positions (kpc).
        v_total: Total circular velocity (km/s).

    Returns:
        Acceleration in km^2 s^-2 kpc^-1 (same units as A0_MOND).
    """
    return v_total**2 / radius


# ---------------------------------------------------------------------------
# Notebook support functions — encapsulate all physics so notebooks stay clean
# ---------------------------------------------------------------------------


def interpolate_v_bary(
    radius_kpc: np.ndarray,
    v_baryon_total: np.ndarray,
    r_query: float,
) -> float:
    """Safely interpolate baryonic velocity at an arbitrary radius.

    Uses the signed-square convention: interpolates V^2_bary (with sign)
    then recovers velocity as sign(x) * sqrt(|x|).  This avoids imaginary
    values when the profile crosses zero (e.g. inner gas depressions).

    Args:
        radius_kpc: Radial positions of the profile (kpc), must be sorted.
        v_baryon_total: Baryonic velocity at each radius (km/s); may be
            negative where gas contribution dominates.
        r_query: Radius at which to evaluate V_bary (kpc).

    Returns:
        Interpolated baryonic velocity in km/s (float).  Returns NaN if
        r_query is outside the profile range.
    """
    if r_query < radius_kpc[0] or r_query > radius_kpc[-1]:
        return float("nan")
    # Signed square: preserves sign information across zero crossings
    v2_signed = np.sign(v_baryon_total) * v_baryon_total**2
    v2_at_r = float(np.interp(r_query, radius_kpc, v2_signed))
    return float(np.sign(v2_at_r) * np.sqrt(abs(v2_at_r)))


def compute_total_model_velocity(
    radius: np.ndarray,
    v_bary: np.ndarray,
    model_name: str,
    param1: float,
    param2: float | None = None,
) -> np.ndarray:
    """Reconstruct the total model velocity from stored database parameters.

    Encapsulates all coupling logic so notebooks never contain inline physics.

    Coupling conventions:
        - NFW:            quadrature  V_total = sqrt(V_bary^2 + V_NFW^2)
        - MOND fixed/free: physics-derived (Simple interpolation function)
        - Rational Taper: additive    V_total = V_bary + omega*R/(1+R/R_t)

    Args:
        radius: Radial positions (kpc).
        v_bary: Baryonic velocity at each radius (km/s).
        model_name: One of "nfw", "mond_fixed", "mond_free", "rational_taper".
        param1: First model parameter as stored in model_fits.param1.
            NFW -> concentration c; MOND -> a0 (km^2/s^2/kpc); RT -> omega (km/s/kpc).
        param2: Second model parameter as stored in model_fits.param2 (optional).
            NFW -> V_200 (km/s); RT -> R_t (kpc); MOND -> None.

    Returns:
        Total model velocity array in km/s.

    Raises:
        ValueError: If model_name is not recognised.
    """
    if model_name == "nfw":
        if param2 is None:
            raise ValueError("NFW requires param2 = V_200")
        v_nfw = nfw_velocity(radius, float(param1), float(param2))
        return np.sqrt(v_bary**2 + v_nfw**2)
    elif model_name in ("mond_fixed", "mond_free"):
        return mond_velocity(radius, v_bary, a0=float(param1))
    elif model_name == "rational_taper":
        if param2 is None:
            raise ValueError("Rational Taper requires param2 = R_t")
        omega = float(param1)
        R_t = float(param2)
        return v_bary + omega * radius / (1.0 + radius / R_t)
    else:
        raise ValueError(f"Unknown model_name: '{model_name}'")


def compute_transition_diagnostics(
    radius_kpc: np.ndarray,
    v_bary_profile: np.ndarray,
    omega: float,
    R_t: float,
) -> dict:
    """Compute physical quantities at the Rational Taper transition radius R_t.

    At R = R_t the taper correction equals omega*R_t / (1 + R_t/R_t) = omega*R_t / 2,
    which is exactly half the saturation velocity V_sat = omega * R_t.

    Uses ``interpolate_v_bary`` to evaluate the baryonic velocity at R_t safely
    (signed-square convention; no imaginary numbers near zero crossings).

    Args:
        radius_kpc: Radial positions of the profile (kpc), sorted ascending.
        v_bary_profile: Baryonic velocity at each radius (km/s).
        omega: Taper slope parameter (km/s/kpc) from model_fits.param1.
        R_t: Transition radius (kpc) from model_fits.param2.

    Returns:
        Dict with keys:
            v_bary_rt   – V_bary interpolated at R_t (km/s)
            v_corr_rt   – Taper correction at R_t = omega*R_t/2 (km/s)
            v_total_rt  – Additive total velocity at R_t (km/s)
            g_obs       – Total centripetal acceleration at R_t (km^2/s^2/kpc)
            g_bary      – Baryonic centripetal acceleration at R_t (km^2/s^2/kpc)
            eta_additive    – g_obs / g_bary (additive coupling)
            eta_quadrature  – hypothetical quadrature g_obs / g_bary for comparison

        All values are NaN if R_t <= 0 or baryonic interpolation fails / g_bary = 0.
    """
    _nan = {
        "v_bary_rt": float("nan"),
        "v_corr_rt": float("nan"),
        "v_total_rt": float("nan"),
        "g_obs": float("nan"),
        "g_bary": float("nan"),
        "eta_additive": float("nan"),
        "eta_quadrature": float("nan"),
    }

    if not (np.isfinite(omega) and np.isfinite(R_t) and R_t > 0):
        return _nan

    v_bary_rt = interpolate_v_bary(radius_kpc, v_bary_profile, R_t)
    if not np.isfinite(v_bary_rt):
        return _nan

    # At R = R_t: correction = omega * R_t / (1 + R_t/R_t) = omega * R_t / 2
    v_corr_rt = omega * R_t / 2.0
    v_total_rt = v_bary_rt + v_corr_rt

    g_obs = v_total_rt**2 / R_t
    g_bary = v_bary_rt**2 / R_t if v_bary_rt != 0 else float("nan")

    if not np.isfinite(g_bary) or g_bary == 0:
        return {**_nan, "v_bary_rt": v_bary_rt, "v_corr_rt": v_corr_rt,
                "v_total_rt": v_total_rt, "g_obs": g_obs}

    eta_additive = g_obs / g_bary
    # Hypothetical quadrature: (V_bary^2 + V_corr^2) / R_t / g_bary
    eta_quadrature = (v_bary_rt**2 + v_corr_rt**2) / R_t / g_bary

    return {
        "v_bary_rt": v_bary_rt,
        "v_corr_rt": v_corr_rt,
        "v_total_rt": v_total_rt,
        "g_obs": g_obs,
        "g_bary": g_bary,
        "eta_additive": eta_additive,
        "eta_quadrature": eta_quadrature,
    }


def compute_validation_metrics(
    radius: np.ndarray,
    v_bary: np.ndarray,
    v_bary_reference: np.ndarray,
    min_radius: float = 2.0,
    threshold_pct: float = 5.0,
) -> "pd.DataFrame":
    """Compare V_bary against a reference and flag deviations.

    Args:
        radius: Radii in kpc.
        v_bary: Our computed V_bary (km/s).
        v_bary_reference: Reference V_bary to compare against (km/s).
        min_radius: Only evaluate radii above this value (kpc).
        threshold_pct: Flag percentage differences above this value.

    Returns:
        DataFrame with columns: radius_kpc, v_bary, v_bary_ref,
        pct_diff, exceeds_threshold.
    """
    import pandas as pd

    mask = radius > min_radius
    r = radius[mask]
    vb = v_bary[mask]
    vb_ref = v_bary_reference[mask]

    pct_diff = np.where(
        vb_ref != 0,
        100.0 * np.abs(vb - vb_ref) / np.abs(vb_ref),
        0.0,
    )

    return pd.DataFrame({
        "radius_kpc": r,
        "v_bary": vb,
        "v_bary_ref": vb_ref,
        "pct_diff": pct_diff,
        "exceeds_threshold": pct_diff > threshold_pct,
    })
