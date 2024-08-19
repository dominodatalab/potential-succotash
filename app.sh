#!/usr/bin/env bash

pip install -r requirements.txt --user

export PYTHONPATH=$(pwd):$PYTHONPATH
python dash_cost_dashboard.py
