"""Check saved full cap runs against their settings and the Gaussian controls."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    report = {"sweeps": {}}
    for sweep,prefix in (("05_height_sweep","H"),("06_radius_sweep","R")):
        base = ROOT / "results"
        cap = base / "spherical_caps" / sweep / "full"
        gaussian = base / "gaussian" / sweep / "full"
        cm = json.loads((cap/f"{prefix}_parameters.json").read_text())
        gm = json.loads((gaussian/f"{prefix}_parameters.json").read_text())
        assert cm["quick"] is False
        for key in ("recursion_depth","line_points","random_vectors","seed","roi_sublattices"):
            assert cm[key] == gm[key], key
        for key,value in gm["base_parameters"].items():
            if key == "profile":
                assert value == "gaussian" and cm["base_parameters"][key] == "spherical_cap"
                continue
            assert cm["base_parameters"][key] == value, key
        with np.load(cap/f"{prefix}_flat_reference.npz") as cf, np.load(gaussian/f"{prefix}_flat_reference.npz") as gf:
            for key in ("energy","position","sites","phases","display_xy"):
                np.testing.assert_allclose(cf[key],gf[key],atol=1e-12)
            flat_error = float(np.max(abs(cf["line_ldos"]-gf["line_ldos"])))
            np.testing.assert_allclose(cf["line_ldos"],gf["line_ldos"],atol=1e-8,rtol=1e-6)
            flat_draws = cf["roi_draws"].copy() if prefix == "H" else None
        files = sorted(cap.glob("H_h*nm.npz" if prefix == "H" else "R_R*nm.npz"))
        assert len(files)==4
        cases=[]
        for path in files:
            with np.load(path) as d:
                assert not bool(d["quick"]) and int(d["recursion_depth"])==1500
                assert int(d["random_vectors"])==8
                for key in ("line_ldos","roi_ldos","roi_error","dos","strain","Bpaper"):
                    assert np.isfinite(d[key]).all(), (path.name,key)
                for key in ("line_ldos","roi_ldos","dos"):
                    assert d[key].min()>=0
                np.testing.assert_allclose(d["roi_draws"].mean(axis=1),d["roi_ldos"])
                paired = d["roi_draws"]-(flat_draws if prefix=="H" else d["flat_roi_draws"])
                np.testing.assert_allclose(paired.mean(axis=1),d["roi_difference"])
                np.testing.assert_allclose(paired.std(axis=1,ddof=1)/np.sqrt(8),d["roi_difference_error"])
                params=json.loads(str(d["parameters_json"]))
                assert params["profile"]=="spherical_cap" and params["lambda_r2_mev"]==0
                cases.append({"file":path.name,"height_nm":params["height"],"radius_nm":params["radius"],
                              "roi_atoms":int(d["roi_count"]),"max_cell_strain":float(d["strain"].max()),
                              "max_abs_Bpaper_T":float(abs(d["Bpaper"]).max())})
        report["sweeps"][sweep]={"matched_gaussian_settings":True,
                                 "flat_line_max_abs_difference":flat_error,"cases":cases}
    output=ROOT/"results/spherical_caps/validation.json"
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
