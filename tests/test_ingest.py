"""Tests for the SPARC data parsers and ingestion pipeline."""

import pytest
import numpy as np
import pandas as pd

from src.ingest import (
    SPARC_COLUMNS,
    extract_galaxy_name_from_filename,
    parse_sparc_rotmod,
)
from src.physics import compute_v_bary


class TestParseSparc:
    def test_returns_correct_columns(self, sample_sparc_file):
        df = parse_sparc_rotmod(sample_sparc_file)
        expected = ["Rad", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"]
        assert list(df.columns) == expected

    def test_skips_comment_lines(self, sample_sparc_file):
        df = parse_sparc_rotmod(sample_sparc_file)
        assert len(df) == 5

    def test_values_are_float(self, sample_sparc_file):
        df = parse_sparc_rotmod(sample_sparc_file)
        for col in df.columns:
            assert df[col].dtype == np.float64

    def test_first_row_values(self, sample_sparc_file):
        df = parse_sparc_rotmod(sample_sparc_file)
        assert df.iloc[0]["Rad"] == pytest.approx(0.50)
        assert df.iloc[0]["Vobs"] == pytest.approx(25.0)
        assert df.iloc[0]["Vgas"] == pytest.approx(10.0)

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_sparc_rotmod(tmp_path / "nonexistent.dat")

    def test_negative_vgas_parsed(self, negative_vgas_file):
        df = parse_sparc_rotmod(negative_vgas_file)
        assert df.iloc[0]["Vgas"] == pytest.approx(-5.0)


class TestExtractGalaxyName:
    def test_standard_name(self):
        assert extract_galaxy_name_from_filename("NGC5055_rotmod.dat") == "NGC5055"

    def test_ugc_name(self):
        assert extract_galaxy_name_from_filename("UGC02885_rotmod.dat") == "UGC02885"

    def test_path_with_directory(self, tmp_path):
        path = tmp_path / "NGC5055_rotmod.dat"
        assert extract_galaxy_name_from_filename(path) == "NGC5055"

    def test_name_without_rotmod(self):
        assert extract_galaxy_name_from_filename("M33.dat") == "M33"


class TestComputeVBaryIngest:
    """Integration tests: parse -> compute V_bary."""

    def test_all_positive_after_parse(self, sample_sparc_file):
        df = parse_sparc_rotmod(sample_sparc_file)
        v_bary = compute_v_bary(
            df["Vgas"].values,
            df["Vdisk"].values,
            df["Vbul"].values,
            upsilon_disk=0.5,
            upsilon_bulge=0.7,
        )
        assert np.all(v_bary >= 0)

    def test_negative_gas_handled(self, negative_vgas_file):
        df = parse_sparc_rotmod(negative_vgas_file)
        v_bary = compute_v_bary(
            df["Vgas"].values,
            df["Vdisk"].values,
            df["Vbul"].values,
        )
        assert np.all(v_bary >= 0)

    def test_upsilon_changes_vbary(self, sample_sparc_file):
        df = parse_sparc_rotmod(sample_sparc_file)

        v_low = compute_v_bary(
            df["Vgas"].values, df["Vdisk"].values, df["Vbul"].values,
            upsilon_disk=0.3,
        )
        v_high = compute_v_bary(
            df["Vgas"].values, df["Vdisk"].values, df["Vbul"].values,
            upsilon_disk=0.8,
        )
        assert np.all(v_high >= v_low)
