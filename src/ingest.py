"""Parsers and ingestion pipeline for SPARC rotation curve data.

Handles:
  - Combined MRT (Machine-Readable Table) files from Lelli et al. (2016)
    containing all 175 galaxies in a single fixed-width file.
  - Individual SPARC _rotmod.dat files (one galaxy per file)
  - The Schneider_2026_SPARC_Fit_Parameters.csv from baryonic-omega-analysis
    (used to seed the database with prior fit results for reference)
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.database import get_session, init_db, insert_galaxy, insert_radial_profiles
from src.physics import HELIUM_FACTOR, compute_v_bary
from src.utils import get_project_root, setup_logger

logger = setup_logger(__name__)

SPARC_COLUMNS = ["Rad", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"]


# ---------------------------------------------------------------------------
# SPARC individual .dat files
# ---------------------------------------------------------------------------


def parse_sparc_rotmod(filepath: str | Path) -> pd.DataFrame:
    """Parse a single SPARC _rotmod.dat file.

    Skips comment lines starting with '!'.

    Args:
        filepath: Path to the .dat file.

    Returns:
        DataFrame with columns: Rad, Vobs, errV, Vgas, Vdisk, Vbul, SBdisk, SBbul.

    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If file has unexpected number of columns.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"SPARC file not found: {filepath}")

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        comment="!",
        names=SPARC_COLUMNS,
        header=None,
    )

    if len(df.columns) != 8:
        raise ValueError(
            f"Expected 8 columns, got {len(df.columns)} in {filepath.name}"
        )

    for col in SPARC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Parsed %s: %d data points", filepath.name, len(df))
    return df


def extract_galaxy_name_from_filename(filepath: str | Path) -> str:
    """Extract galaxy name from SPARC filename convention.

    Examples:
        'NGC5055_rotmod.dat' -> 'NGC5055'
        'UGC02885_rotmod.dat' -> 'UGC02885'
    """
    name = Path(filepath).stem
    if "_rotmod" in name:
        name = name.split("_rotmod")[0]
    return name


def ingest_sparc_file(
    filepath: str | Path,
    galaxy_name: Optional[str] = None,
    upsilon_disk: float = 0.5,
    upsilon_bulge: float = 0.7,
    db_path: Optional[str] = None,
) -> str:
    """Full pipeline: parse SPARC file -> compute V_bary -> insert into DB."""
    filepath = Path(filepath)
    raw_df = parse_sparc_rotmod(filepath)
    galaxy_id = galaxy_name or extract_galaxy_name_from_filename(filepath)

    v_bary = compute_v_bary(
        raw_df["Vgas"].values,
        raw_df["Vdisk"].values,
        raw_df["Vbul"].values,
        upsilon_disk=upsilon_disk,
        upsilon_bulge=upsilon_bulge,
    )

    profiles_df = pd.DataFrame({
        "radius_kpc": raw_df["Rad"],
        "v_obs": raw_df["Vobs"],
        "v_err": raw_df["errV"],
        "v_gas": raw_df["Vgas"],
        "v_disk": raw_df["Vdisk"],
        "v_bulge": raw_df["Vbul"],
        "v_baryon_total": v_bary,
    })

    engine = init_db(db_path)
    session = get_session(engine)
    try:
        insert_galaxy(session, galaxy_id)
        n_rows = insert_radial_profiles(session, galaxy_id, profiles_df)
        logger.info(
            "Ingested %s: %d profiles (Upsilon_d=%.2f, Upsilon_b=%.2f)",
            galaxy_id, n_rows, upsilon_disk, upsilon_bulge,
        )
    finally:
        session.close()

    return galaxy_id


# ---------------------------------------------------------------------------
# Combined MRT file (all 175 galaxies)
# ---------------------------------------------------------------------------


def parse_massmodels_mrt(filepath: str | Path) -> dict[str, pd.DataFrame]:
    """Parse the combined MassModels MRT file (Lelli et al. 2016, Table 2).

    This file contains rotation curve data for all 175 SPARC galaxies in a
    single fixed-width file. Format per the MRT header:
        Col  1-11: Galaxy ID (A11)
        Col 13-18: Distance (F6.2, Mpc)
        Col 20-25: Radius (F6.2, kpc)
        Col 27-32: Vobs (F6.2, km/s)
        Col 34-38: errV (F5.2, km/s)
        Col 40-45: Vgas (F6.2, km/s)
        Col 47-52: Vdisk (F6.2, km/s)
        Col 54-59: Vbul (F6.2, km/s)
        Col 61-67: SBdisk (F7.2, solLum/pc2)
        Col 69-76: SBbul (F8.2, solLum/pc2)

    Args:
        filepath: Path to MassModels_Lelli2016c.mrt file.

    Returns:
        Dict mapping galaxy_id -> DataFrame with standard SPARC columns.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"MRT file not found: {filepath}")

    with open(filepath, "r") as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("---"):
            data_start = i + 1

    records = []
    for line in lines[data_start:]:
        if len(line.strip()) == 0:
            continue
        try:
            galaxy_id = line[0:11].strip()
            rad = float(line[19:25])
            vobs = float(line[26:32])
            errv = float(line[33:38])
            vgas = float(line[39:45])
            vdisk = float(line[46:52])
            vbul = float(line[53:59])
            sbdisk = float(line[60:67])
            sbbul = float(line[68:76])
            records.append({
                "galaxy_id": galaxy_id,
                "Rad": rad,
                "Vobs": vobs,
                "errV": errv,
                "Vgas": vgas,
                "Vdisk": vdisk,
                "Vbul": vbul,
                "SBdisk": sbdisk,
                "SBbul": sbbul,
            })
        except (ValueError, IndexError) as e:
            logger.warning("Skipping malformed line %d: %s", data_start + 1, e)

    all_df = pd.DataFrame(records)
    logger.info(
        "Parsed MRT file: %d data points across %d galaxies",
        len(all_df),
        all_df["galaxy_id"].nunique(),
    )

    result = {}
    for gid, group in all_df.groupby("galaxy_id", sort=False):
        result[gid] = group[SPARC_COLUMNS].reset_index(drop=True)

    return result


def parse_sparc_metadata_mrt(filepath: str | Path) -> pd.DataFrame:
    """Parse the SPARC galaxy metadata MRT file (Lelli et al. 2016, Table 1).

    Whitespace-separated with 19 fields per line:
        [0] Galaxy, [1] T (Hubble type), [2] D (Mpc), [3] e_D, [4] f_D,
        [5] Inc (deg), [6] e_Inc, [7] L[3.6] (10^9 Lsun), [8] e_L,
        [9] Reff (kpc), [10] SBeff, [11] Rdisk (kpc), [12] SBdisk,
        [13] MHI (10^9 Msun), [14] RHI (kpc), [15] Vflat (km/s),
        [16] e_Vflat, [17] Q (quality flag), [18] References

    Returns:
        DataFrame with columns: galaxy_id, distance_mpc, inclination,
        luminosity_band_36, r_disk_kpc, sb_disk, v_flat, quality_flag.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Metadata MRT file not found: {filepath}")

    with open(filepath, "r") as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("---"):
            data_start = i + 1

    records = []
    for line in lines[data_start:]:
        if len(line.strip()) == 0:
            continue
        try:
            fields = line.split()
            if len(fields) < 18:
                continue
            records.append({
                "galaxy_id": fields[0],
                "distance_mpc": float(fields[2]),
                "inclination": float(fields[5]),
                "luminosity_band_36": float(fields[7]),
                "r_disk_kpc": float(fields[11]),
                "sb_disk": float(fields[12]),
                "v_flat": float(fields[15]),
                "quality_flag": int(fields[17]),
            })
        except (ValueError, IndexError) as e:
            logger.warning("Skipping metadata line: %s", e)

    df = pd.DataFrame(records)
    logger.info("Parsed metadata for %d galaxies", len(df))
    return df


def ingest_massmodels_mrt(
    massmodels_path: str | Path,
    metadata_path: Optional[str | Path] = None,
    upsilon_disk: float = 0.5,
    upsilon_bulge: float = 0.7,
    db_path: Optional[str] = None,
) -> list[str]:
    """Ingest all galaxies from the combined MassModels MRT file.

    Optionally also ingests metadata from the SPARC Table 1 MRT file.

    Args:
        massmodels_path: Path to MassModels_Lelli2016c.mrt.
        metadata_path: Optional path to SPARC_Lelli2016c.mrt for metadata.
        upsilon_disk: Mass-to-light ratio for disk. Default 0.5.
        upsilon_bulge: Mass-to-light ratio for bulge. Default 0.7.
        db_path: Optional database path override.

    Returns:
        List of galaxy_ids that were ingested.
    """
    galaxies_data = parse_massmodels_mrt(massmodels_path)

    metadata_df = None
    if metadata_path:
        metadata_df = parse_sparc_metadata_mrt(metadata_path)
        metadata_df = metadata_df.set_index("galaxy_id")

    engine = init_db(db_path)
    session = get_session(engine)

    ingested = []
    try:
        for galaxy_id, raw_df in galaxies_data.items():
            v_bary = compute_v_bary(
                raw_df["Vgas"].values,
                raw_df["Vdisk"].values,
                raw_df["Vbul"].values,
                upsilon_disk=upsilon_disk,
                upsilon_bulge=upsilon_bulge,
            )

            profiles_df = pd.DataFrame({
                "radius_kpc": raw_df["Rad"],
                "v_obs": raw_df["Vobs"],
                "v_err": raw_df["errV"],
                "v_gas": raw_df["Vgas"],
                "v_disk": raw_df["Vdisk"],
                "v_bulge": raw_df["Vbul"],
                "v_baryon_total": v_bary,
            })

            meta_kwargs = {}
            if metadata_df is not None and galaxy_id in metadata_df.index:
                row = metadata_df.loc[galaxy_id]
                meta_kwargs = {
                    "distance_mpc": float(row["distance_mpc"]),
                    "inclination": float(row["inclination"]),
                    "luminosity_band_36": float(row["luminosity_band_36"]),
                    "r_disk_kpc": float(row["r_disk_kpc"]),
                    "sb_disk": float(row["sb_disk"]),
                    "v_flat": float(row["v_flat"]),
                    "quality_flag": int(row["quality_flag"]),
                }

            insert_galaxy(session, galaxy_id, **meta_kwargs)
            insert_radial_profiles(session, galaxy_id, profiles_df)
            ingested.append(galaxy_id)

        logger.info(
            "Ingested %d galaxies from MRT (Upsilon_d=%.2f, Upsilon_b=%.2f)",
            len(ingested), upsilon_disk, upsilon_bulge,
        )
    finally:
        session.close()

    return ingested


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest SPARC rotation curve data")
    parser.add_argument("--mrt", type=str, help="Path to MassModels MRT file (all galaxies)")
    parser.add_argument("--metadata", type=str, help="Path to SPARC metadata MRT file")
    parser.add_argument("--file", type=str, help="Path to single _rotmod.dat file")
    parser.add_argument("--name", type=str, help="Galaxy name override (single file only)")
    parser.add_argument("--upsilon-disk", type=float, default=0.5)
    parser.add_argument("--upsilon-bulge", type=float, default=0.7)
    args = parser.parse_args()

    if args.mrt:
        ingest_massmodels_mrt(
            args.mrt,
            metadata_path=args.metadata,
            upsilon_disk=args.upsilon_disk,
            upsilon_bulge=args.upsilon_bulge,
        )
    elif args.file:
        ingest_sparc_file(
            args.file,
            galaxy_name=args.name,
            upsilon_disk=args.upsilon_disk,
            upsilon_bulge=args.upsilon_bulge,
        )
    else:
        parser.print_help()
