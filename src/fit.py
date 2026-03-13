"""CLI fitting pipeline: run all four models on SPARC galaxies and store results.

Each galaxy gets four model fits stored as separate rows in the model_fits table:
    nfw, mond_fixed, mond_free, rational_taper

Usage:
    python src/fit.py --galaxy NGC3198 --plot
    python src/fit.py --all --quality 1
    python src/fit.py --all --model rational_taper --force
    python src/fit.py --galaxy NGC3198 --force --plot
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from src.database import (
    Galaxy,
    ModelFit,
    delete,
    get_engine,
    get_session,
    init_db,
    insert_model_fit,
    query_fits_as_dataframe,
    query_profiles_as_dataframe,
)
from src.physics import (
    A0_MOND,
    compute_v_bary,
    fit_mond_fixed,
    fit_mond_free,
    fit_nfw,
    fit_rational_taper,
    mond_velocity,
    nfw_velocity,
)
from src.colors import MODEL_COLORS, OBS_COLORS
from src.utils import get_project_root, setup_logger

logger = setup_logger(__name__)

ALL_MODELS = ["nfw", "mond_fixed", "mond_free", "rational_taper"]
METHOD_VERSION = "v1"
UPSILON_DISK = 0.5
UPSILON_BULGE = 0.7


# ---------------------------------------------------------------------------
# Single-galaxy fitting
# ---------------------------------------------------------------------------


def run_fits_for_galaxy(
    galaxy_id: str,
    session,
    models: Optional[list] = None,
    method_version: str = METHOD_VERSION,
    upsilon_disk: float = UPSILON_DISK,
    upsilon_bulge: float = UPSILON_BULGE,
    force: bool = False,
    plot: bool = False,
    output_dir: Optional[str] = None,
) -> dict:
    """Fit all specified models to one galaxy and store results in the database.

    Args:
        galaxy_id: SPARC galaxy identifier (e.g. "NGC3198").
        session: Active SQLAlchemy session.
        models: List of model names to fit. Defaults to all four models.
        method_version: Tag stored in model_fits.method_version.
        upsilon_disk: Mass-to-light ratio for disk component.
        upsilon_bulge: Mass-to-light ratio for bulge component.
        force: If True, delete existing fit rows and refit.
        plot: If True, generate a four-model comparison plot.
        output_dir: Override directory for plot output.

    Returns:
        Dict mapping model_name -> ModelFitResult for converged fits.
    """
    if models is None:
        models = ALL_MODELS

    # ---- load profile data ------------------------------------------------
    df = query_profiles_as_dataframe(session, galaxy_id)
    if df.empty:
        logger.error("%s: no profile data found, skipping", galaxy_id)
        return {}

    radius = df["radius_kpc"].values.astype(float)
    v_obs = df["v_obs"].values.astype(float)
    v_gas = df["v_gas"].values.astype(float)
    v_disk = df["v_disk"].values.astype(float)
    v_bulge = df["v_bulge"].values.astype(float)

    # Coerce nullable v_err column: NaN → 0.0 (physics._safe_errors floors zeros)
    raw_err = df["v_err"].values
    v_err = np.nan_to_num(raw_err.astype(float), nan=0.0)

    # Use stored v_baryon_total when available; recompute if any NaNs
    raw_vbary = df["v_baryon_total"].values.astype(float)
    if np.any(~np.isfinite(raw_vbary)):
        logger.warning(
            "%s: v_baryon_total has NaN entries, recomputing from components",
            galaxy_id,
        )
        v_bary = compute_v_bary(v_gas, v_disk, v_bulge, upsilon_disk, upsilon_bulge)
    else:
        v_bary = raw_vbary

    # ---- skip-check -------------------------------------------------------
    if force:
        already_fitted: set = set()
    else:
        existing = query_fits_as_dataframe(session, galaxy_id=galaxy_id)
        already_fitted = set(existing["model_name"].values) if not existing.empty else set()

    # ---- dispatch table ---------------------------------------------------
    def _dispatch(model_name: str):
        kwargs = dict(
            galaxy_id=galaxy_id,
            method_version=method_version,
            upsilon_disk=upsilon_disk,
            upsilon_bulge=upsilon_bulge,
        )
        if model_name == "nfw":
            return fit_nfw(radius, v_obs, v_err, v_bary, **kwargs)
        elif model_name == "mond_fixed":
            return fit_mond_fixed(radius, v_obs, v_err, v_bary, **kwargs)
        elif model_name == "mond_free":
            return fit_mond_free(radius, v_obs, v_err, v_bary, **kwargs)
        elif model_name == "rational_taper":
            return fit_rational_taper(radius, v_obs, v_err, v_bary, **kwargs)
        else:
            raise ValueError(f"Unknown model: '{model_name}'")

    # ---- fit each model ---------------------------------------------------
    results: dict = {}
    for model_name in models:
        if model_name not in ALL_MODELS:
            logger.warning("Unknown model '%s', skipping", model_name)
            continue
        if model_name in already_fitted:
            logger.info(
                "%s [%s]: already fitted, skipping (use --force to refit)",
                galaxy_id,
                model_name,
            )
            continue

        # Delete existing row if forcing
        if force:
            session.execute(
                delete(ModelFit)
                .where(ModelFit.galaxy_id == galaxy_id)
                .where(ModelFit.model_name == model_name)
            )
            session.flush()

        try:
            result = _dispatch(model_name)
            insert_model_fit(session, result.to_dict())
            results[model_name] = result
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "%s [%s]: unexpected exception — %s", galaxy_id, model_name, exc
            )
            continue

    # ---- optional plot ----------------------------------------------------
    if plot and results:
        out_dir = output_dir or str(
            get_project_root() / "results" / "figures" / "rotation_curves"
        )
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = str(Path(out_dir) / f"{galaxy_id}_comparison.png")
        generate_comparison_plot(
            galaxy_id=galaxy_id,
            radius=radius,
            v_obs=v_obs,
            v_err=v_err,
            v_gas=v_gas,
            v_disk=v_disk,
            v_bulge=v_bulge,
            v_bary=v_bary,
            results=results,
            upsilon_disk=upsilon_disk,
            upsilon_bulge=upsilon_bulge,
            output_path=out_path,
        )

    return results


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------


def generate_comparison_plot(
    galaxy_id: str,
    radius: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    v_bary: np.ndarray,
    results: dict,
    upsilon_disk: float = UPSILON_DISK,
    upsilon_bulge: float = UPSILON_BULGE,
    output_path: Optional[str] = None,
) -> None:
    """Generate a four-model comparison rotation curve plot.

    All converged models are overlaid on the observed data. Baryonic
    components are shown as dashed reference lines.

    Args:
        galaxy_id: Galaxy name (used for title and default filename).
        radius: Radial positions (kpc).
        v_obs: Observed rotation velocities (km/s).
        v_err: Velocity uncertainties (km/s).
        v_gas, v_disk, v_bulge: Raw SPARC component velocities (km/s).
        v_bary: Total baryonic velocity (km/s).
        results: Dict of model_name -> ModelFitResult from run_fits_for_galaxy.
        upsilon_disk: Disk mass-to-light ratio (for component scaling).
        upsilon_bulge: Bulge mass-to-light ratio.
        output_path: Full path for saved PNG. If None, uses project default.
    """
    import matplotlib.pyplot as plt

    r_plot = np.linspace(radius.min(), radius.max(), 300)

    _MODEL_STYLE = {
        "nfw":            ("NFW",             MODEL_COLORS["nfw"],            2.0),
        "mond_fixed":     ("MOND (fixed a₀)", MODEL_COLORS["mond_fixed"],     2.0),
        "mond_free":      ("MOND (free a₀)",  MODEL_COLORS["mond_free"],      2.0),
        "rational_taper": ("Rational Taper",   MODEL_COLORS["rational_taper"], 2.5),
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    # Observed data
    ax.errorbar(
        radius, v_obs, yerr=np.where(v_err > 0, v_err, np.nan),
        fmt="ko", ms=4, capsize=2, lw=0.8, label=r"$V_\mathrm{obs}$", zorder=5,
    )

    # Baryonic components (dashed reference lines)
    ax.plot(radius, np.abs(v_gas), "g--", lw=1.2, alpha=0.7, label=r"$|V_\mathrm{gas}|$")
    ax.plot(
        radius,
        np.sqrt(upsilon_disk) * np.abs(v_disk),
        "r--", lw=1.2, alpha=0.7,
        label=rf"$\sqrt{{\Upsilon_d}}|V_\mathrm{{disk}}|$",
    )
    if np.any(v_bulge != 0):
        ax.plot(
            radius,
            np.sqrt(upsilon_bulge) * np.abs(v_bulge),
            "m-.", lw=1.2, alpha=0.7,
            label=rf"$\sqrt{{\Upsilon_b}}|V_\mathrm{{bulge}}|$",
        )
    ax.plot(radius, v_bary, color=OBS_COLORS["v_bary"], ls="-", lw=1.8, alpha=0.85, label=r"$V_\mathrm{bary}$")

    # Model curves — reconstruct on fine grid from stored parameters
    bic_lines = []
    for model_name in ALL_MODELS:
        if model_name not in results:
            continue
        res = results[model_name]
        if not res.converged:
            continue
        label_base, color, lw = _MODEL_STYLE[model_name]

        # Reconstruct v_model on fine grid using stored params
        p1 = res.param1
        p2 = res.param2
        v_bary_fine = np.interp(r_plot, radius, v_bary)

        if model_name == "nfw":
            v_model = np.sqrt(v_bary_fine**2 + nfw_velocity(r_plot, p1, p2)**2)
        elif model_name in ("mond_fixed", "mond_free"):
            v_model = mond_velocity(r_plot, v_bary_fine, a0=float(p1))
        else:  # rational_taper
            v_model = v_bary_fine + p1 * r_plot / (1.0 + r_plot / p2)

        bic_val = res.bic if res.bic is not None else float("nan")
        label = f"{label_base} (BIC={bic_val:.1f})"
        ax.plot(r_plot, v_model, color=color, lw=lw, label=label)
        bic_lines.append(f"{label_base}: {bic_val:.1f}")

    ax.set_xlabel("Radius (kpc)", fontsize=12)
    ax.set_ylabel("Velocity (km/s)", fontsize=12)
    ax.set_title(f"{galaxy_id} — Four-Model Comparison", fontsize=13)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()

    if output_path is None:
        out_dir = get_project_root() / "results" / "figures" / "rotation_curves"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{galaxy_id}_comparison.png")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison plot: %s", output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit NFW, MOND, and Rational Taper models to SPARC galaxies."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--galaxy", metavar="ID", help="Fit a single galaxy by ID")
    target.add_argument("--all", action="store_true", help="Fit all galaxies in the DB")

    parser.add_argument(
        "--quality", type=int, choices=[1, 2, 3],
        help="Filter galaxies by quality flag (used with --all)",
    )
    parser.add_argument(
        "--model", choices=ALL_MODELS,
        help="Fit only this model (default: all four)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete and refit even if results already exist",
    )
    parser.add_argument("--plot", action="store_true", help="Save comparison plots")
    parser.add_argument("--output-dir", help="Override figure output directory")
    parser.add_argument(
        "--method", default=METHOD_VERSION,
        help=f"Method version tag (default: {METHOD_VERSION})",
    )
    parser.add_argument(
        "--upsilon-disk", type=float, default=UPSILON_DISK,
        help=f"Disk mass-to-light ratio (default: {UPSILON_DISK})",
    )
    parser.add_argument(
        "--upsilon-bulge", type=float, default=UPSILON_BULGE,
        help=f"Bulge mass-to-light ratio (default: {UPSILON_BULGE})",
    )
    args = parser.parse_args()

    models = [args.model] if args.model else None

    engine = init_db()
    session = get_session(engine)

    try:
        if args.galaxy:
            results = run_fits_for_galaxy(
                galaxy_id=args.galaxy,
                session=session,
                models=models,
                method_version=args.method,
                upsilon_disk=args.upsilon_disk,
                upsilon_bulge=args.upsilon_bulge,
                force=args.force,
                plot=args.plot,
                output_dir=args.output_dir,
            )
            converged = [m for m, r in results.items() if r.converged]
            bic_str = ", ".join(
                f"{m}={results[m].bic:.1f}" for m in converged if results[m].bic is not None
            )
            logger.info("%s done. Converged: %s. BICs: %s", args.galaxy, converged, bic_str)

        else:  # --all
            # Collect galaxy IDs, optionally filtered by quality_flag
            query = session.query(Galaxy.galaxy_id)
            if args.quality is not None:
                query = query.filter(Galaxy.quality_flag == args.quality)
            galaxy_ids = [row[0] for row in query.all()]

            if not galaxy_ids:
                logger.error("No galaxies found (quality filter: %s)", args.quality)
                sys.exit(1)

            logger.info("Fitting %d galaxies (%s models each)...", len(galaxy_ids),
                        len(models) if models else "all 4")

            # Use tqdm if available, else plain loop with periodic logging
            try:
                from tqdm import tqdm
                galaxy_iter = tqdm(galaxy_ids, desc="Fitting", unit="galaxy")
            except ImportError:
                galaxy_iter = galaxy_ids

            converge_counts: dict = {m: 0 for m in (models or ALL_MODELS)}
            for i, gid in enumerate(galaxy_iter):
                if not hasattr(galaxy_iter, "set_description") and i % 10 == 0:
                    logger.info("Progress: %d / %d", i, len(galaxy_ids))
                try:
                    res = run_fits_for_galaxy(
                        galaxy_id=gid,
                        session=session,
                        models=models,
                        method_version=args.method,
                        upsilon_disk=args.upsilon_disk,
                        upsilon_bulge=args.upsilon_bulge,
                        force=args.force,
                        plot=args.plot,
                        output_dir=args.output_dir,
                    )
                    for m, r in res.items():
                        if r.converged:
                            converge_counts[m] = converge_counts.get(m, 0) + 1
                except Exception as exc:  # noqa: BLE001
                    logger.error("%s: top-level exception — %s", gid, exc)

            logger.info("=== Convergence summary ===")
            for m, cnt in converge_counts.items():
                logger.info("  %s: %d / %d converged", m, cnt, len(galaxy_ids))

    finally:
        session.close()


if __name__ == "__main__":
    main()
