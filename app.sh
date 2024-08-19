#!/usr/bin/env bash

pip install -r requirements.txt --user

export PYTHONPATH=$(pwd):$PYTHONPATH
python domino_cost/cost_dashboard.py
