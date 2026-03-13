"""End-to-end integration tests for the full pipeline."""

import pytest
import numpy as np

from src.database import (
    get_session,
    init_db,
    insert_model_fit,
    query_fits_as_dataframe,
    query_profiles_as_dataframe,
)
from src.ingest import ingest_sparc_file, parse_sparc_rotmod
from src.physics import compute_v_bary, fit_mond_fixed, fit_nfw, fit_rational_taper


class TestIngestAndQuery:
    def test_ingest_round_trip(self, sample_sparc_file, tmp_path):
        """Verify ingested data matches original file values."""
        db_path = str(tmp_path / "test_rt.db")
        original = parse_sparc_rotmod(sample_sparc_file)

        galaxy_id = ingest_sparc_file(sample_sparc_file, db_path=db_path)

        engine = init_db(db_path)
        session = get_session(engine)
        from_db = query_profiles_as_dataframe(session, galaxy_id)
        session.close()

        np.testing.assert_allclose(
            from_db["radius_kpc"].values, original["Rad"].values, rtol=1e-10,
        )
        np.testing.assert_allclose(
            from_db["v_obs"].values, original["Vobs"].values, rtol=1e-10,
        )
        np.testing.assert_allclose(
            from_db["v_gas"].values, original["Vgas"].values, rtol=1e-10,
        )

    def test_negative_vgas_preserved(self, negative_vgas_file, tmp_path):
        """Negative V_gas values should survive the ingest round-trip."""
        db_path = str(tmp_path / "test_neg.db")
        galaxy_id = ingest_sparc_file(negative_vgas_file, db_path=db_path)

        engine = init_db(db_path)
        session = get_session(engine)
        df = query_profiles_as_dataframe(session, galaxy_id)
        session.close()

        assert df.iloc[0]["v_gas"] == pytest.approx(-5.0)
        v_bary = compute_v_bary(
            df["v_gas"].values, df["v_disk"].values, df["v_bulge"].values,
        )
        assert np.all(v_bary >= 0)


class TestFullPipeline:
    def test_ingest_fit_store_query(self, sample_sparc_file, tmp_path):
        """Full pipeline: ingest -> fit all models -> store -> query back."""
        db_path = str(tmp_path / "test_full.db")

        galaxy_id = ingest_sparc_file(
            sample_sparc_file,
            upsilon_disk=0.5,
            upsilon_bulge=0.7,
            db_path=db_path,
        )

        engine = init_db(db_path)
        session = get_session(engine)

        df = query_profiles_as_dataframe(session, galaxy_id)
        assert len(df) == 5

        v_bary = compute_v_bary(
            df["v_gas"].values, df["v_disk"].values, df["v_bulge"].values,
            upsilon_disk=0.5, upsilon_bulge=0.7,
        )

        radius = df["radius_kpc"].values
        v_obs = df["v_obs"].values
        v_err = df["v_err"].values

        # Fit all models
        r_nfw = fit_nfw(radius, v_obs, v_err, v_bary, galaxy_id=galaxy_id)
        r_mond = fit_mond_fixed(radius, v_obs, v_err, v_bary, galaxy_id=galaxy_id)
        r_taper = fit_rational_taper(radius, v_obs, v_err, v_bary, galaxy_id=galaxy_id)

        # Store results
        for result in [r_nfw, r_mond, r_taper]:
            if result.converged:
                insert_model_fit(session, result.to_dict())

        # Query back
        fits_df = query_fits_as_dataframe(session, galaxy_id=galaxy_id)
        session.close()

        assert len(fits_df) >= 2  # at least MOND (always converges) + taper
        assert "rational_taper" in fits_df["model_name"].values
        assert "mond_fixed" in fits_df["model_name"].values

    def test_bic_stored_correctly(self, sample_sparc_file, tmp_path):
        """BIC values should survive the DB round-trip."""
        db_path = str(tmp_path / "test_bic.db")
        galaxy_id = ingest_sparc_file(sample_sparc_file, db_path=db_path)

        engine = init_db(db_path)
        session = get_session(engine)
        df = query_profiles_as_dataframe(session, galaxy_id)

        v_bary = compute_v_bary(
            df["v_gas"].values, df["v_disk"].values, df["v_bulge"].values,
        )

        result = fit_mond_fixed(
            df["radius_kpc"].values, df["v_obs"].values, df["v_err"].values,
            v_bary, galaxy_id=galaxy_id,
        )

        insert_model_fit(session, result.to_dict())

        fits_df = query_fits_as_dataframe(session, galaxy_id=galaxy_id, model_name="mond_fixed")
        session.close()

        assert len(fits_df) == 1
        assert fits_df.iloc[0]["bic"] == pytest.approx(result.bic, rel=1e-6)
