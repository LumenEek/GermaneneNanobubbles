"""Execute these project's plain-Python notebooks without a networked kernel.

Usage: python run_notebooks.py notebooks/spherical_caps/05_*.ipynb
Records executed code, stdout and figures in each notebook; stops on any error.
Only the %matplotlib inline magic is translated, to the noninteractive Agg backend.
"""
from pathlib import Path
import argparse
import base64
import contextlib
import datetime
import hashlib
import io
import json
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Tee(io.StringIO):
    def write(self, s):
        sys.__stdout__.write(s)
        sys.__stdout__.flush()
        return super().write(s)


def run(path):
    path = path.resolve()
    notebook = json.loads(path.read_text())
    namespace = {"__name__": "__main__"}
    os.chdir(path.parent)
    started = time.perf_counter()
    print(f"START {path.name}", flush=True)
    count = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        source = "".join(cell["source"])
        source = source.replace("%matplotlib inline", "# Agg backend selected by runner")
        outputs = []

        def show(*args, **kwargs):
            for number in plt.get_fignums():
                fig = plt.figure(number)
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", dpi=100)
                outputs.append({"output_type": "display_data", "metadata": {},
                                "data": {"image/png": base64.b64encode(buffer.getvalue()).decode()}})
            plt.close("all")

        plt.show = show
        log = Tee()
        cell["execution_count"] = count
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                exec(compile(source, f"{path.name}:cell{count}", "exec"), namespace)
        except Exception as exc:
            outputs.append({"output_type": "error", "ename": type(exc).__name__,
                            "evalue": str(exc), "traceback": traceback.format_exc().splitlines()})
            raise
        finally:
            if log.getvalue():
                outputs.insert(0, {"output_type": "stream", "name": "stdout",
                                   "text": log.getvalue().splitlines(keepends=True)})
            cell["outputs"] = outputs
            path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
        print(f"  cell {count} finished ({time.perf_counter()-started:.1f} s elapsed)", flush=True)
    notebook.setdefault("metadata", {})["local_execution"] = {
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seconds": time.perf_counter()-started,
        "python": sys.version,
        "quick": os.environ.get("HB_QUICK", "0") == "1",
        "model_sha256": hashlib.sha256((ROOT / "model.py").read_bytes()).hexdigest(),
        "runner": "run_notebooks.py (isolated process, Agg, no kernel server)"}
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"DONE {path.name}: {time.perf_counter()-started:.1f} s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    run(args.notebook)
