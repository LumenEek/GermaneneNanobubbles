# Germanene nanobubble calculations

Tight-binding calculations for Gaussian bubbles and spherical caps in germanene,
using TBPLaS: periodic-sheet sweeps and zigzag ribbons with periodic boundaries
along their length and open boundaries across their width. Lengths are nm; energies
are eV.

The repository contains the shared model, seven calculation notebooks, validation
tests, figure-export scripts and tools for running notebooks and configuring Git.

## Active notebooks

| Folder | Notebook | Calculation |
| --- | --- | --- |
| `notebooks/gaussian/` | `05_radius3_height_sweep.ipynb` | Gaussian: R=3 nm; h=0.5, 1, 1.5, 2 nm |
| `notebooks/gaussian/` | `06_height1_radius_sweep.ipynb` | Gaussian: h=1 nm; R=2, 3, 4, 5 nm |
| `notebooks/spherical_caps/` | `05_radius3_height_sweep.ipynb` | Cap: same height sweep |
| `notebooks/spherical_caps/` | `06_height1_radius_sweep.ipynb` | Cap: same radius sweep |
| `notebooks/ribbons_partial_obc/` | `07_gaussian_partial_obc.ipynb` | Gaussian ribbon, h=1.5 nm and R=2.5 nm |
| `notebooks/ribbons_partial_obc/` | `08_spherical_cap_partial_obc.ipynb` | Cap ribbon, h=1.5 nm and R=2.5 nm |
| `notebooks/ribbons_partial_obc/` | `09_gaussian_size_matching.ipynb` | Wider Gaussian ribbons matched to cap width or geometric volume |

The periodic sweeps use the Lanczos/continued-fraction solver in `model.py`.

## Model and conventions

All active calculations use intrinsic SOC 100 meV and zero intrinsic Rashba.
`model.py` constructs the lattice, normal-directed buckling, strain-dependent
nearest-neighbour hopping, PMF and spectral estimators. Periodic-sheet display
coordinates are X=-y, Y=x. Spectral means are per spin orbital; spatial ribbon
LDOS maps sum both spins per atom.

For a Gaussian, R=3 sigma; for a cap, R is the footprint radius. Equal values do
not imply equal widths. The exact cap has sphere radius S=(R²+h²)/(2h), height
sqrt(S²-r²)-(S-h) inside r<R and zero outside. Require 0<=h<R. Its slope jumps
at the flat-sheet join. There is no elastic relaxation, rim smoothing or
strain-dependent SOC.

Periodic sweeps use 1201 energies on [-1,1] eV, 10 meV Lorentzian HWHM, 1500
recursion steps and eight random-phase vectors (seed 20260923). ROIs include
both sublattices; uncertainty bands are one stochastic standard error. Height
sweeps use 9,360 spin orbitals; radius sweeps share a 25,800-orbital cell.

Full ribbons use 1,500 atoms, 81 Bloch momenta, 1201 energies on [-0.3,0.3] eV
and Gaussian broadening with standard deviation 5 meV. Spatial maps are at
-50, 0 and +50 meV. Notebook 09 keeps the cap height, center and ribbon fixed:
R_G=4.892823401 nm matches nominal half-height width; R_G=4.079380611 nm matches
geometric volume below the surface within the ribbon, including edge truncation
and periodic images. These are geometric comparisons, not mechanical constraints.

## Setup

Full calculations used Python 3.12.9. TBPLaS has compiled extensions; source
installation needs a C/Fortran toolchain and CMake. Publication plots use
Matplotlib's LaTeX rendering, requiring LaTeX and `dvipng`. The environment is
not included in this repository.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python3 tools/setup_git.py
```

`requirements-tested.txt` records the exact direct package versions used. It is
not a full cross-platform lockfile. SciPy remains below 1.14 because TBPLaS 1.6
imports `scipy.integrate.trapz`. Compiler configuration may be platform-specific;
installation has not been repeated on a clean machine.

After each clone, run `python3 tools/setup_git.py`. This enables the notebook
clean filter: local notebooks retain executed plots and logs, while Git stores
source and documentation without outputs. The filter uses only the standard
library. Without this local setup, Git does not strip outputs automatically.

## Reproduce the calculations

From the repository root, with the environment activated, run the independent
periodic sweeps:

```bash
python run_notebooks.py notebooks/gaussian/05_radius3_height_sweep.ipynb
python run_notebooks.py notebooks/gaussian/06_height1_radius_sweep.ipynb
python run_notebooks.py notebooks/spherical_caps/05_radius3_height_sweep.ipynb
python run_notebooks.py notebooks/spherical_caps/06_height1_radius_sweep.ipynb
```

Run both base ribbons before notebook 09, which reads their full results:

```bash
python run_notebooks.py notebooks/ribbons_partial_obc/07_gaussian_partial_obc.ipynb
python run_notebooks.py notebooks/ribbons_partial_obc/08_spherical_cap_partial_obc.ipynb
python run_notebooks.py notebooks/ribbons_partial_obc/09_gaussian_size_matching.ipynb
```

The runner executes one notebook in a separate process, limits BLAS/OpenMP
threads, and saves outputs and provenance. Alternatively use `jupyter lab` and a
fresh kernel per notebook. `HB_QUICK=1` reduces sampling for notebooks 05–08;
outputs go into separate `quick/` directories and are not manuscript data.
Notebook 09 requires full base results and does not support quick mode.

Results go under `results/<profile>/.../full/` and
`results/ribbons_partial_obc/.../full/`. Arrays, plots, metadata and CSV summaries
are generated when the notebooks run and excluded from Git.

## Validation and figure export

```bash
python -m unittest discover -s tests -v
# After all four full periodic sweeps:
python validate_results.py
python export_manuscript_figures.py
# After full ribbon notebooks 07 and 08:
python ribbon_manuscript.py
```

Tests cover cap geometry, Hamiltonian boundary structure and the spectral solver
against an exact resolvent. Saved-result checks cover full-run settings, flat
references, spectral normalization and stochastic errors. Notebook 09 validates
its new ribbon spectra and integrated state weights.

Figure scripts export to `results/manuscript_figures/` by default. Set
`HB_FIGURE_DIR` to an absolute directory path to choose another export destination.

## Version control

The `.gitattributes` file and `tools/clean_notebook.py` keep notebook source in
Git without embedded execution outputs. Configure the filter with
`python3 tools/setup_git.py`. Review `git diff --cached` before committing changes.
