"""Export verified full spherical-cap sweep figures to a configurable directory."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
from figure_export import FIGURE_DIR, figure_destination

ROOT = Path(__file__).resolve().parent
FIGURES = ("height_profiles", "geometry_pmf", "line_ldos_maps",
           "local_spectra", "roi_spectra", "roi_difference_dos")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = {"model_sha256": digest(ROOT / "model.py"),
                "boundary_conditions": "periodic x and y", "sweeps": [], "figures": []}
    for sweep, prefix in (("05_height_sweep", "H"), ("06_radius_sweep", "R")):
        source = ROOT / "results/spherical_caps" / sweep / "full"
        metadata = json.loads((source / f"{prefix}_parameters.json").read_text())
        assert metadata["quick"] is False
        assert metadata["recursion_depth"] == 1500 and metadata["random_vectors"] == 8
        assert metadata["base_parameters"]["profile"] == "spherical_cap"
        assert metadata["base_parameters"]["lambda_r2_mev"] == 0
        arrays = sorted(source.glob(f"{prefix}_h*nm.npz" if prefix == "H" else "R_R*nm.npz"))
        assert len(arrays) == 4
        for path in arrays:
            with np.load(path) as data:
                assert not bool(data["quick"])
                assert int(data["recursion_depth"]) == 1500
                for key in ("height", "strain", "Bpaper", "line_ldos", "roi_ldos", "dos"):
                    assert np.isfinite(data[key]).all(), (path.name,key)
                for key in ("line_ldos", "roi_ldos", "dos"):
                    assert data[key].min() >= 0, (path.name,key)
        manifest["sweeps"].append({"source": str(source.relative_to(ROOT)),
                                    "settings": metadata,
                                    "arrays": {f.name:digest(f) for f in arrays}})
        for name in FIGURES:
            original = source / f"{prefix}_{name}.png"
            destination = figure_destination(f"SC_{prefix}_{name}.png")
            shutil.copy2(original, destination)
            manifest["figures"].append({"source": str(original.relative_to(ROOT)),
                                         "destination": destination.name,
                                         "sha256": digest(destination)})
    (ROOT / "results/spherical_caps/manuscript_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(f"Copied {len(manifest['figures'])} full-run figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
