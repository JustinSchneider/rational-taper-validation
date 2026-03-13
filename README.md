# Rational Taper Validation

**Empirical Validation of the Rational Taper Kinematic Model: A Comparative Analysis Against ΛCDM and MOND**

_Justin Schneider — Independent Researcher_

_Manuscript under consideration for peer review_

---

## Overview

This repository contains the full analysis pipeline for a head-to-head BIC model comparison across 175 SPARC galaxies, pitting the **Rational Taper** (RT) kinematic model against NFW (ΛCDM), Fixed MOND, and Free MOND.

The Rational Taper model predicts circular velocity as:

$$V_\mathrm{model} = V_\mathrm{bary} + \frac{\omega R}{1 + R/R_t}$$

where $\omega$ and $R_t$ are the two free parameters and $V_\mathrm{bary}$ is the baryonic velocity from SPARC mass decomposition. No dark matter halo is assumed.

---

## Key Findings

- **Statistical parity with NFW:** RT wins lowest BIC in 47.4% of galaxies (NFW 39.4%); median ΔBIC(RT − NFW) = +0.6 across the full sample — competitive without a dark matter halo.
- **g₀/2 symmetry:** The empirical median total acceleration at the transition radius $R_t$ is 6.51 × 10⁻¹¹ m/s², within 8.5% of $a_0/2$ — an unexpected alignment with the MOND acceleration scale.
- **Iso-luminosity tracks:** $g(R_t)$ vs. $R_t$ forms diagonal $1/R$ tracks stratified by baryonic mass, with fitted slope 0.514 (consistent with the BTFR prediction of 0.5).
- **Cusp-core advantage:** RT inner-disk median residual −2.49 km/s vs. NFW −4.73 km/s; RT has nearly half the cusp-core bias.
- **Additive form signature:** RAR consistency ratio $\eta_\mathrm{add}/\eta_\mathrm{quad} \approx 2$, distinguishing the additive structure from quadrature alternatives.

---

## Repository Layout

```
├── notebooks/          # Analysis notebooks (run in order; see guide below)
├── src/                # Python source: physics, fitting, database, ingestion
├── data/
│   ├── raw/            # SPARC MRT files + seeded RT fit parameters
│   └── processed/
│       ├── galaxy_dynamics.db          # SQLite: 175 galaxies × 4 models
│       └── supplemental/               # Public supplemental CSVs
├── results/
│   ├── figures/        # 24 publication-ready figures
│   └── tables/         # Robustness/sensitivity CSV outputs
├── tests/              # pytest suite (physics, ingest, integration)
├── METHODOLOGY.md      # Full methodology reference
└── DATA_DICTIONARY.md  # Column descriptions for supplemental CSVs
```

---

## Notebook Guide

Run notebooks in order. All physics and fitting logic lives in `src/`; notebooks consume it.

| Notebook                                                                         | What it does                                                                                                    |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| [00_setup.ipynb](notebooks/00_setup.ipynb)                                       | Initialize database, ingest SPARC data, seed RT fits from CSV, fit NFW and MOND models — **start here**         |
| [01_database_validation.ipynb](notebooks/01_database_validation.ipynb)           | Sample overview, convergence audit, representative rotation curves                                              |
| [02_bic_tournament.ipynb](notebooks/02_bic_tournament.ipynb)                     | BIC win counts, ΔBIC histograms, hard-case rotation curve comparisons                                           |
| [03_spatial_residuals.ipynb](notebooks/03_spatial_residuals.ipynb)               | Cusp-core test, stacked radial residuals for all four models                                                    |
| [04_geometric_transitions.ipynb](notebooks/04_geometric_transitions.ipynb)       | g₀/2 symmetry analysis, iso-luminosity tracks, scaling relations                                                |
| [05_additive_coupling.ipynb](notebooks/05_additive_coupling.ipynb)               | RAR consistency ratio, additive vs. quadrature model structure comparison                                       |
| [06_due_diligence.ipynb](notebooks/06_due_diligence.ipynb)                       | Four robustness tests: $R_t$ artifact check, SB independence, baryon-fraction residuals, quality stratification |
| [07_presubmission_validation.ipynb](notebooks/07_presubmission_validation.ipynb) | Convergence audit, mass-to-light sensitivity, NFW cross-checks                                                  |
| [08_supplemental_data_prep.ipynb](notebooks/08_supplemental_data_prep.ipynb)     | Generate supplemental CSVs and LaTeX tables for publication                                                     |

---

## Data

**Raw inputs** (not redistributed; copy from SPARC public release):

- `data/raw/MassModels_Lelli2016c.mrt` — rotation curves and mass decomposition
- `data/raw/SPARC_Lelli2016c.mrt` — galaxy metadata
- `data/raw/Schneider_2026_SPARC_Fit_Parameters.csv` — seeded RT fit parameters

**Processed outputs** (generated by Notebook 00):

- `data/processed/galaxy_dynamics.db` — SQLite database, 175 galaxies × 4 models (700 fits)

**Public supplemental data** (linked from manuscript):

- `data/processed/supplemental/Tournament_Results.csv` — one row per galaxy; BIC, fit parameters, and diagnostics for all four models
- `data/processed/supplemental/Radial_Residuals.csv` — one row per radial data point (~3,400 total); model velocities and residuals

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for full column descriptions.

---

## Setup

Python 3.9+ required.

```bash
pip install -r requirements.txt
```

Then open `notebooks/00_setup.ipynb` and **Run All**.

This initializes the database, ingests SPARC data, seeds the Rational Taper fits, and fits NFW and MOND models for all 175 galaxies.

```bash
# Run tests
pytest tests/

# Fit a single galaxy with a comparison plot (CLI alternative)
python src/fit.py --galaxy NGC3198 --plot
```

---

## Further Reading

- [METHODOLOGY.md](METHODOLOGY.md) — detailed methodology: model equations, fitting conventions, BIC framework, database schema
- [DATA_DICTIONARY.md](DATA_DICTIONARY.md) — supplemental CSV column descriptions and physical constants
