#!/usr/bin/env bash
# Create the isolated Python 3.11 environment used by the LightFM baseline.
#
# LightFM 1.17 ships a C extension that does not build on recent Python
# interpreters, and its setup.py sets __builtins__.__LIGHTFM_SETUP__, which
# fails under modern build backends where __builtins__ is a dict. It is
# therefore built in a dedicated venv from a lightly patched sdist. The main
# pipeline calls into this env through scripts/lightfm_bridge.py whenever the
# LightFM baseline is enabled.
#
# Usage:  bash scripts/setup_lightfm_env.sh [PYTHON311]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY311="${1:-python3.11}"
VENV="$ROOT/.venv-lightfm"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo ">> Creating venv at $VENV with $PY311"
"$PY311" -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip setuptools wheel Cython "numpy<3" scipy scikit-learn pandas

echo ">> Fetching and patching LightFM 1.17 sdist"
cd "$WORK"
"$VENV/bin/python" - <<'PY'
import urllib.request, json
meta = json.load(urllib.request.urlopen("https://pypi.org/pypi/lightfm/1.17/json"))
url = next(f["url"] for f in meta["urls"] if f["packagetype"] == "sdist")
urllib.request.urlretrieve(url, "lightfm.tar.gz")
PY
tar xzf lightfm.tar.gz
# Patch the one line that breaks under dict-form __builtins__.
python3 - <<'PY'
import pathlib
p = pathlib.Path("lightfm-1.17/setup.py")
s = p.read_text()
s = s.replace(
    "__builtins__.__LIGHTFM_SETUP__ = True",
    "import builtins as _b\n_b.__LIGHTFM_SETUP__ = True",
)
p.write_text(s)
PY

echo ">> Installing patched LightFM into the venv"
"$VENV/bin/python" -m pip install --quiet --no-build-isolation "$WORK/lightfm-1.17"

echo ">> Verifying"
"$VENV/bin/python" -c "from lightfm import LightFM; print('LightFM OK')"
echo ">> Done."
