"""Enable the notebook clean filter in this repository's local Git configuration."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    top = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=ROOT, text=True).strip()
    if Path(top).resolve() != ROOT:
        raise SystemExit('Initialize Git in GermaneneNanobubbleTBPLaS itself, not a parent folder.')
    for key, value in [('filter.notebook-clean.clean', 'python3 tools/clean_notebook.py'),
                       ('filter.notebook-clean.required', 'true')]:
        subprocess.run(['git', 'config', '--local', key, value], cwd=ROOT, check=True)
    print('Notebook filter enabled: outputs stay on disk and are omitted from Git.')


if __name__ == '__main__':
    main()
