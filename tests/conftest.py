"""Shared pytest fixtures for the tapered model comparison test suite."""

import pytest
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base, get_session


@pytest.fixture
def sample_sparc_content():
    """Return string content of a minimal valid SPARC _rotmod.dat file."""
    return (
        "! Galaxy: TEST001\n"
        "! Columns: Rad Vobs errV Vgas Vdisk Vbul SBdisk SBbul\n"
        "  0.50   25.0   3.0   10.0   15.0   0.0   100.0   0.0\n"
        "  1.00   50.0   4.0   20.0   30.0   5.0    80.0   2.0\n"
        "  2.00   80.0   5.0   25.0   40.0   8.0    50.0   1.5\n"
        "  3.00  100.0   5.0   28.0   45.0  10.0    30.0   1.0\n"
        "  5.00  110.0   6.0   30.0   48.0  10.0    15.0   0.5\n"
    )


@pytest.fixture
def sample_sparc_file(sample_sparc_content, tmp_path):
    """Write sample SPARC content to a temporary .dat file."""
    filepath = tmp_path / "TEST001_rotmod.dat"
    filepath.write_text(sample_sparc_content)
    return filepath


@pytest.fixture
def negative_vgas_content():
    """SPARC data where V_gas is negative at inner radii."""
    return (
        "! Galaxy: TEST_NEGGAS\n"
        "  0.50   25.0   3.0   -5.0   20.0   0.0   100.0   0.0\n"
        "  1.00   50.0   4.0   10.0   30.0   0.0    80.0   0.0\n"
        "  2.00   80.0   5.0   25.0   40.0   0.0    50.0   0.0\n"
    )


@pytest.fixture
def negative_vgas_file(negative_vgas_content, tmp_path):
    """Write negative V_gas content to a temporary file."""
    filepath = tmp_path / "TEST_NEGGAS_rotmod.dat"
    filepath.write_text(negative_vgas_content)
    return filepath


@pytest.fixture
def in_memory_engine():
    """Create an in-memory SQLite engine with schema initialized."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def in_memory_session(in_memory_engine):
    """Create a session bound to the in-memory database."""
    factory = sessionmaker(bind=in_memory_engine)
    session = factory()
    yield session
    session.close()


@pytest.fixture
def synthetic_rotation_curve():
    """Generate a synthetic rotation curve with known parameters for testing.

    Returns dict with: radius, v_gas, v_disk, v_bulge, v_bary, v_obs, v_err,
    and the true omega/R_t used to generate the Rational Taper data.
    """
    rng = np.random.default_rng(42)
    radius = np.linspace(1.0, 10.0, 20)

    v_gas = 20.0 * np.sqrt(radius / 5.0)
    v_disk = 80.0 * np.sqrt(radius / (radius + 2.0))
    v_bulge = np.zeros_like(radius)

    # Known taper parameters
    true_omega = 5.0   # km/s/kpc
    true_rt = 6.0      # kpc

    v_bary = np.sqrt(v_gas**2 + v_disk**2)

    # V_obs = V_bary + tapered correction + noise
    v_corr = true_omega * radius / (1.0 + radius / true_rt)
    noise = rng.normal(0, 2.0, size=len(radius))
    v_obs = v_bary + v_corr + noise
    v_err = np.full_like(radius, 3.0)

    return {
        "radius": radius,
        "v_gas": v_gas,
        "v_disk": v_disk,
        "v_bulge": v_bulge,
        "v_bary": v_bary,
        "v_obs": v_obs,
        "v_err": v_err,
        "true_omega": true_omega,
        "true_rt": true_rt,
    }
