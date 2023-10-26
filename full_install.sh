#!/usr/bin/env bash
set -euo pipefail
python3 -m pip install -e common/[tests]
python3 -m pip install -e core/[all,tests]
python3 -m pip install -e features/
python3 -m pip install -e tabular/[all,tests]
python3 -m pip install -e timeseries/[all,tests]
python3 -m pip install -e autogluon/
