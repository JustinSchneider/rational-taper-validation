# Methodology: Tapered Model Comparison

**Author:** Justin Schneider (independent researcher)
**Last Updated:** March 2026
**Status:** Data processing complete, analyzing results

---

## 1. Scientific Background

This project extends the work of Schneider (2026, submitted) by situating the **Rational Taper** model in a rigorous comparative framework against the two dominant paradigms in galactic dynamics:

- **Λ CDM**: The NFW dark matter halo profile (Navarro, Frenk & White 1996)
- **MOND**: Modified Newtonian Dynamics with the Simple interpolation function (Milgrom 1983; Famaey & Binney 2005)

The central question is not merely whether the Rational Taper _fits_ well, but whether it is _statistically competitive_ with established models on an equal-footing comparison using the Bayesian Information Criterion (BIC), and what physical insight the comparison provides.

The Rational Taper result from the prior project (baryonic-omega-analysis) is the starting point: the model was preferred over the linear omega correction in 74.3% of the 171-galaxy SPARC clean sample. This project places that model into a four-way tournament against NFW and MOND.

---

## 2. Dataset

### SPARC Database — Lelli, McGaugh, & Schombert (2016)

- **Paper:** "SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves" (AJ, 152, 157)
- **Local files:** `data/raw/MassModels_Lelli2016c.mrt` (rotation curves), `data/raw/SPARC_Lelli2016c.mrt` (metadata)
- **Clean sample:** 171 galaxies after applying quality filters (inclination, minimum data points, convergence)
- **Key fields:** `Rad` (kpc), `Vobs` (km/s), `errV`, `Vgas`, `Vdisk`, `Vbul`, `SBdisk`, `SBbul`
- **Note:** `Vdisk` and `Vbul` are provided at $\Upsilon = 1$; mass-to-light ratios are applied during analysis.

### Prior Results CSV

- **File:** `data/raw/Schneider_2026_SPARC_Fit_Parameters.csv`
- **Source:** baryonic-omega-analysis Phase III full catalog results
- **Content:** Linear and Rational Taper fit parameters for all 171 galaxies (omega, R_t, BIC, RMSE, convergence flags)
- **Use:** Reference baseline; initial database can be seeded from this file to avoid re-fitting the Rational Taper from scratch.

---

## 3. Baryonic Velocity Computation

All four models use an identical baryonic mass decomposition — the key methodological requirement for a fair comparison.

$$V_{bary}(R) = \sqrt{|V_{gas}| \cdot V_{gas} + \Upsilon_d \cdot |V_{disk}| \cdot V_{disk} + \Upsilon_b \cdot |V_{bulge}| \cdot V_{bulge}}$$

The $|V| \cdot V$ sign-preserving convention handles negative SPARC velocity contributions (e.g., central gas depressions) without producing imaginary numbers (Lelli et al. 2016, Eq. 2).

**Fixed mass-to-light ratios (same for all models):**

| Parameter | Symbol             | Value |
| --------- | ------------------ | ----- |
| Disk M/L  | $\Upsilon_{disk}$  | 0.5   |
| Bulge M/L | $\Upsilon_{bulge}$ | 0.7   |

**Implementation:** `src/physics.py :: compute_v_bary()`

---

## 4. The Four Models

### 4.1 Model A — NFW Halo (Λ CDM)

**Free parameters: 2** ($c$, $V_{200}$)

$$V_{NFW}^2(R) = V_{200}^2 \frac{\ln(1 + cx) - cx/(1+cx)}{x \left[\ln(1+c) - c/(1+c)\right]}$$

where $x = R/R_{200}$ and $R_{200} = V_{200} / (10 H_0)$ with $H_0 = 73$ km/s/Mpc.

**Total model:** $V_{total} = \sqrt{V_{bary}^2 + V_{NFW}^2}$ (quadrature — NFW is an independent gravitational potential)

**Fitting details:**

- Multi-start optimizer: 4 initial conditions varying $(c, V_{200})$; retain lowest-$\chi^2$ solution
- Bounds: $c \in [1, 100]$, $V_{200} \in [1, 2000]$ km/s
- `scipy.optimize.curve_fit` with `absolute_sigma=True`

**Implementation:** `src/physics.py :: fit_nfw()`

---

### 4.2 Model B — Fixed MOND

**Free parameters: 0** (zero-parameter prediction)

Uses the Famaey & Binney (2005) "Simple" interpolation function, $\mu(x) = x/(1+x)$, yielding:

$$V_{MOND}^2 = V_{bary}^2 + \frac{V_{bary}^2}{2} \left(\sqrt{1 + \frac{4 a_0 R}{V_{bary}^2}} - 1\right)$$

with canonical $a_0 = 1.2 \times 10^{-10}$ m/s$^2$ $= 3703$ km$^2$ s$^{-2}$ kpc$^{-1}$ (fixed).

**No optimization is performed.** Chi-squared is computed directly from the MOND prediction. Since $k = 0$, BIC = $\chi^2$ exactly.

**Degeneracy note:** Where $V_{bary} = 0$, $V_{MOND} = 0$ by definition (no baryons → no gravitational source).

**Implementation:** `src/physics.py :: fit_mond_fixed()`

---

### 4.3 Model B′ — Free MOND

**Free parameters: 1** ($a_0$ free)

Same interpolation function as Fixed MOND, but $a_0$ is allowed to vary as a proxy for distance uncertainty and baryonic modeling systematics. Bounds: $a_0 \in [1000, 10000]$ km$^2$ s$^{-2}$ kpc$^{-1}$ (spanning ~0.27–2.7 times the canonical value).

**Implementation:** `src/physics.py :: fit_mond_free()`

---

### 4.4 Model C — Rational Taper (Schneider 2026)

**Free parameters: 2** ($\omega$, $R_t$)

$$V_{model}(R) = V_{bary}(R) + \frac{\omega \cdot R}{1 + R/R_t}$$

At small $R$: correction $\to \omega R$ (linear).
At large $R$: correction $\to \omega R_t = V_{sat}$ (constant — flat rotation curve recovery).

**Additive coupling** (not quadrature), consistent with the baryonic-omega-analysis methodology and physically distinct from dark matter halo models.

**Fitting details:**

- Multi-start optimizer: 4 initial conditions; retain lowest-$\chi^2$ solution
- Bounds: $\omega \in [0, 200]$ km/s/kpc, $R_t \in [0.1, 5 R_{max}]$ kpc
- Derived quantity: $V_{sat} = \omega \cdot R_t$ (not a fit parameter; computed from best-fit $\omega$ and $R_t$)

**Implementation:** `src/physics.py :: fit_rational_taper()`

---

## 5. Model Selection: BIC

$$\text{BIC} = \chi^2 + k \ln(n)$$

where $\chi^2$ is the total (non-reduced) weighted chi-squared, $k$ is the number of free parameters, and $n$ is the number of data points. Lower BIC is preferred.

| Model          | $k$ | BIC penalty at $n=15$ |
| -------------- | --- | --------------------- |
| Fixed MOND     | 0   | 0                     |
| Free MOND      | 1   | 2.71                  |
| NFW            | 2   | 5.42                  |
| Rational Taper | 2   | 5.42                  |

**Interpretation scale** (Kass & Raftery 1995):

| $    | \Delta\text{BIC}     | $   | Evidence |
| ---- | -------------------- | --- | -------- |
| < 2  | Not worth mentioning |
| 2–6  | Positive             |
| 6–10 | Strong               |
| > 10 | Very strong          |

The BIC strongly penalizes NFW and the Rational Taper relative to Fixed MOND. For the Rational Taper to be preferred over Fixed MOND, it must reduce $\chi^2$ by more than $2 \ln(n)$ — a substantial hurdle that validates the result when it occurs.

**Implementation:** `src/physics.py :: compute_bic()`

---

## 6. Key Analyses

### 6.1 BIC Tournament

Apply all four models to each of the 171 SPARC clean-sample galaxies. Report:

- Per-galaxy winner by lowest BIC
- Catalog-level breakdown (% galaxies where each model wins)
- Median $\Delta$BIC between models
- Distribution of $\Delta$BIC (histogram)

### 6.2 Cusp-Core Test

Stack residuals $(V_{obs} - V_{NFW})$ as a function of normalized radius $R/R_d$ across all galaxies. A systematic positive bias at small $R/R_d$ (inner disk) would directly demonstrate the cusp-core problem in the SPARC sample.

### 6.3 Radial Residuals

Stack $(V_{obs} - V_{model})$ vs. $R/R_d$ for all four models side by side. Visualizes where each model systematically fails across the catalog.

### 6.4 g₀/2 Symmetry

For each Rational Taper fit, compute the total centripetal acceleration at $R_t$:

$$g(R_t) = \frac{V_{total}(R_t)^2}{R_t}$$

Test whether the median of $g(R_t)$ across 171 galaxies equals $a_0/2 = 1851.5$ km$^2$ s$^{-2}$ kpc$^{-1}$. The Rational Taper reaches exactly 50% of its asymptotic limit at $R_t$, so if $R_t$ corresponds to the half-saturation point of vacuum polarization, this specific value is predicted.

### 6.5 Iso-Luminosity Tracks

Plot $g_{total}(R_t)$ vs. $R_t$ for all galaxies. The prediction is that galaxies of similar baryonic mass trace parallel $1/R$ diagonal lines (iso-luminosity tracks), rejecting a purely flat universal acceleration threshold.

Preliminary results from this analysis are available in Notebook 11 of baryonic-omega-analysis (described in `docs/internal/board.md` there).

### 6.6 Due Diligence & Systematics (Notebook 06)

Four robustness tests stress-test the Rational Taper findings against common confounds:

| Test                               | Threat                                                | Pass Condition                                                                                                                        |
| ---------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **A: $R_t$ vs $R_{\rm max}$**      | $R_t$ pushed to data boundary by optimizer            | Majority of $R_t < R_{\rm max}$; median $R_t/R_{\rm max} \ll 1$                                                                       |
| **B: Mass/SB Dependence of g(Rₜ)** | $g(R_t) \approx a_0/2$ is a coincidental average      | After removing expected power-law mass trend, log-g residuals show no further trend ($p > 0.05$); log(g) vs SB also flat ($p > 0.05$) |
| **C: Baryon Fraction Residuals**   | Additive coupling fails at high or low $f_{\rm bary}$ | RT per-galaxy slope not significantly worse than NFW (Welch $p > 0.05$)                                                               |
| **D: Quality Stratification**      | BIC wins concentrated in low-quality (Q=3) galaxies   | Q=1 win rate not significantly below overall rate (one-sided binomial test, $p > 0.05$)                                               |
| **E: Predictors of Model Preference** | RT preference is contingent on galaxy type (not general) | Spearman correlations of $\Delta$BIC(NFW−RT) vs. galaxy properties; SB independence ($p > 0.05$) required as positive result; mass/extent correlations characterize boundary conditions |

**Test B methodology — log-scale regression and residual test:** A raw linear regression of $g(R_t)$ vs. mass will always find a significant slope because both quantities span orders of magnitude and $g \propto V^2/R$ naturally scales with baryonic mass (this is the same physics underlying the iso-luminosity tracks in Notebook 04). The scientifically correct test fits a power-law $\log g = \alpha \log M + \beta$ and then tests whether the **residuals** of that fit show any further trend with mass ($p > 0.05$). The power-law index $\alpha$ is compared to the BTFR prediction of $\approx 0.5$. A separate log($g$) vs. SB linear regression is also reported. The figure has three panels: (1) log($g$) vs SB with trend line, (2) log($g$) vs log($M$) with power-law fit, (3) log-$g$ residuals vs log($M$) after power-law removed.

**Test C methodology — within-galaxy demeaning and comparative slope test:** Fractional residuals $(V_{\rm obs} - V_{\rm model})/V_{\rm obs}$ are computed per galaxy and **demeaned within each galaxy** before pooling, removing per-galaxy fit-quality offsets. A per-galaxy slope of the demeaned residual vs. $f_{\rm bary}$ is then computed for both NFW and RT. Both models are expected to show a mild negative slope — this is a structural consequence of fitting a global model to a radially varying rotation curve where the outer flat region anchors the fit, causing slight over-prediction of inner baryon-dominated points. The correct null hypothesis is therefore not "slope = 0" but "RT slope is no worse than NFW slope." A Welch's t-test on the per-galaxy slope distributions tests this: if RT's slopes are not significantly more negative than NFW's ($p > 0.05$), the additive coupling is validated as performing at least as well as the dark-matter benchmark across the baryon fraction gradient.

**Test D methodology — binomial proportion test:** A raw win-rate comparison (Q=1 rate vs. overall rate) is insufficient because even a sub-1% drop over N≈99 galaxies is well within binomial sampling noise. A one-sided binomial test is used instead: H₀ is that the Q=1 win probability equals the overall rate; the test checks whether the observed Q=1 count is significantly below the expectation ($p > 0.05$ = PASS).

**Test E methodology — multivariate Spearman correlation:** Spearman's $\rho$ is computed between $\Delta$BIC(NFW−RT) and five predictors across all 175 galaxies: $\log(M_{\rm bar})$, disk scale length $R_d$, central surface brightness $\mu_0$, and radial extent ratio $R_{\rm max}/R_d$, plus quality-flag breakdown. The sign convention is positive = RT preferred. Significant negative $\rho$ indicates NFW is preferred in that galaxy subpopulation; a null result ($p > 0.05$) for a predictor indicates RT performance is independent of that property. Results (PASS = all four predictors assessed and interpreted):

| Predictor | $\rho$ | $p$ | Interpretation |
|---|---|---|---|
| $\log(M_{\rm bar})$ | −0.316 | <0.001 | NFW preferred above $\sim 10^{10.5}\,M_\odot$; structural boundary condition of RT |
| $R_d$ | −0.221 | 0.01 | NFW preferred for physically larger disks (correlated with mass) |
| $\mu_0$ | +0.035 | 0.72 | **No trend** — RT performance is surface-brightness-independent (positive result) |
| $R_{\rm max}/R_d$ | −0.221 | 0.01 | NFW preferred when curves extend beyond $\sim$10 disk scale lengths |

Quality-flag breakdown: Q=1 (N=99) median $\Delta$BIC = +1.2, win rate 46.5%; Q=2 (N=64) median = +1.1, win rate 53.1%; Q=3 (N=12) median = −0.4, win rate 25.0%. The Q=3 degradation is the anti-overfitting signal: a model exploiting noise would win *more* in noisy data, not less.

### 6.7 Additive vs. Quadrature — RAR Consistency Ratio

At $R_t$, evaluate the Radial Acceleration Relation (RAR) consistency ratio:

$$\eta = \frac{g_{obs}(R_t)}{g_{bary}(R_t)} = \frac{V_{obs}^2(R_t)/R_t}{V_{bary}^2(R_t)/R_t}$$

The additive form yields $\eta > 1$ because $(V_{\rm bary} + V_{\rm corr})^2 > V_{\rm bary}^2$; the cross-term $2V_{\rm bary}V_{\rm corr}$ is absent in any quadrature (independent-field) theory. The observed median $\eta_{\rm additive} \approx 3.6$ and $\eta_{\rm quadrature} \approx 1.8$ reflect the actual distribution of correction amplitudes at $R_t$ across the SPARC sample; the ratio add/quad $\approx 2$ quantifies how much larger the additive cross-term contribution is relative to the quadrature-only term.

---

## 7. Database Schema

The SQLite database (`data/processed/galaxy_dynamics.db`) has three tables:

- **`galaxies`** — Metadata: distance, inclination, luminosity, disk scale length, surface brightness, quality flag
- **`radial_profiles`** — Per-galaxy radial data: $R$, $V_{obs}$, $V_{err}$, $V_{gas}$, $V_{disk}$, $V_{bulge}$, $V_{bary}$
- **`model_fits`** — One row per (galaxy, model): BIC, $\chi^2$, RMSE, convergence flag, two generic parameter columns

The `model_fits` table uses generic `param1`/`param2` columns whose semantics depend on `model_name`:

| `model_name`     | `param1`                     | `param2`         |
| ---------------- | ---------------------------- | ---------------- |
| `nfw`            | $c$ (concentration)          | $V_{200}$ (km/s) |
| `mond_fixed`     | $a_0$ (fixed value, not fit) | —                |
| `mond_free`      | best-fit $a_0$               | —                |
| `rational_taper` | $\omega$ (km/s/kpc)          | $R_t$ (kpc)      |

**Implementation:** `src/database.py`

---

## 8. Fitting Conventions

- **Optimizer:** `scipy.optimize.curve_fit` with `absolute_sigma=True` (Levenberg-Marquardt)
- **Error handling:** Zero/negative errors replaced with minimum nonzero error (floor: 1.0 km/s)
- **Multi-start:** NFW and Rational Taper use 4 initial conditions; retain lowest-$\chi^2$ solution
- **Convergence:** `curve_fit` RuntimeError → `converged=False`, result stored with `nan` parameters
- **Degrees of freedom:** $\text{dof} = n - k$ (where $k$ = number of free parameters)

---

## 9. Software Architecture

```
Raw Data (MRT files) → src/ingest.py → SQLite DB (data/processed/)
                                              ↓
                        src/fit.py (run_fits_for_galaxy)
                          calls → src/physics.py (fit_nfw, fit_mond_*, fit_rational_taper)
                          stores → src/database.py (insert_model_fit)
                                              ↓
                        notebooks/ (query_fits_as_dataframe → analyze → plot)
```

**Core rule:** No physics logic in notebooks. All model equations live in `src/physics.py`. Notebooks call `src` functions only and never implement models inline.

### 9.1 Fitting Pipeline (`src/fit.py`)

`fit.py` provides a CLI fitting pipeline and an importable function for use in notebooks.

**Core function:**

```python
run_fits_for_galaxy(galaxy_id, session, models=None, method_version="v1",
                    upsilon_disk=0.5, upsilon_bulge=0.7,
                    force=False, plot=False, output_dir=None) -> dict
```

- Queries radial profiles from the DB; guards against NaN `v_err` and `v_baryon_total`
- Dispatches to `fit_nfw()`, `fit_mond_fixed()`, `fit_mond_free()`, or `fit_rational_taper()` as needed
- Stores each result via `insert_model_fit(session, result.to_dict())`
- Skip-check: models already fitted are skipped unless `force=True` (which deletes and refits)
- Optionally saves a four-model comparison PNG to `results/figures/rotation_curves/`

**CLI usage:**

```bash
python src/fit.py --galaxy NGC3198 --plot          # Single galaxy, all 4 models
python src/fit.py --all --quality 1                # All Q=1 galaxies
python src/fit.py --all --model nfw --force        # Refit NFW for all galaxies
```

**Rational Taper seeding:** For initial setup, `notebooks/00_setup.ipynb` reads prior Rational Taper parameters from `data/raw/Schneider_2026_SPARC_Fit_Parameters.csv` and seeds them directly into `model_fits` (tagged `method_version="v1_seeded"`). This avoids re-fitting 171 galaxies when results already exist from `baryonic-omega-analysis`.

### 9.2 Notebook-Support Physics Functions (`src/physics.py`)

Three helper functions were added to enforce the no-physics-in-notebooks rule:

| Function                                                                   | Purpose                                                                                         |
| -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `interpolate_v_bary(radius_kpc, v_baryon_total, r_query)`                  | Safe baryonic velocity interpolation using signed-square convention                             |
| `compute_total_model_velocity(radius, v_bary, model_name, param1, param2)` | Reconstruct model velocity from stored DB parameters                                            |
| `compute_transition_diagnostics(radius_kpc, v_bary_profile, omega, R_t)`   | All physical quantities at $R_t$: $g_{obs}$, $g_{bary}$, $\eta_{additive}$, $\eta_{quadrature}$ |

The `interpolate_v_bary` function interpolates the **signed square** of $V_{bary}$ before converting back to velocity. This correctly handles profiles that cross zero (e.g. inner gas depressions) without producing imaginary numbers.

---

## 10. Analysis Notebooks

All notebooks live in `notebooks/`. Run them in order after executing `00_setup.ipynb`.

| Notebook                         | Title                         | Key Outputs                                                                                                                         |
| -------------------------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `00_setup.ipynb`                 | Database Setup & Fitting      | `galaxy_dynamics.db` populated; all 4 models fitted                                                                                 |
| `01_database_validation.ipynb`   | Sample Overview               | Verification table; 6-panel representative rotation curves                                                                          |
| `02_bic_tournament.ipynb`        | Act I: BIC Tournament         | Win counts; ΔBIC histograms (`02_bic_delta_histograms.png`)                                                                         |
| `03_spatial_residuals.ipynb`     | Act I: Spatial Failures       | Cusp-core figure; 4-panel stacked residuals (`03_*.png`)                                                                            |
| `04_geometric_transitions.ipynb` | Act II: g₀/2 & Iso-luminosity | $g(R_t)$ histogram; iso-luminosity scatter (`04_*.png`)                                                                             |
| `05_additive_coupling.ipynb`     | Act III: Theoretical Bridge   | RAR consistency histograms; $\eta$ vs $g_{bary}$ scatter (`05_*.png`)                                                               |
| `06_due_diligence.ipynb`                 | Systematics & Robustness      | Four stress tests: $R_t$ artifact check, SB independence, baryon-fraction residuals (demeaned), quality stratification (`06_*.png`) |
| `07_presubmission_validation.ipynb`      | Pre-submission Validation     | Convergence auditing, Υ sensitivity, NFW cross-checks, morphological controls (`07_*.png`) |
| `08_supplemental_data_prep.ipynb`        | Supplemental Data Prep        | LaTeX table generation for appendix (Tournament_Results, Radial_Residuals) |

All figures are saved as PNG to `results/figures/`.

## 11. References

1. Famaey, B. & Binney, J. (2005). MNRAS, 363, 603. "Modified Newtonian Dynamics in the Milky Way."
2. Famaey, B. & McGaugh, S. S. (2012). LRR, 15, 10. "Modified Newtonian Dynamics (MOND)."
3. Kass, R. E. & Raftery, A. E. (1995). JASA, 90, 773. "Bayes Factors."
4. Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016). AJ, 152, 157. "SPARC: Mass Models for 175 Disk Galaxies."
5. Milgrom, M. (1983). ApJ, 270, 365. "A modification of the Newtonian dynamics."
6. Navarro, J. F., Frenk, C. S., & White, S. D. M. (1996). ApJ, 462, 563. "The Structure of Cold Dark Matter Halos."
7. Schwarz, G. (1978). Annals of Statistics, 6, 461. "Estimating the Dimension of a Model."
8. Schneider, J. (2026). "A Rational Taper Model for Galaxy Rotation Curves." (Submitted, Ap&SS)
