#!/bin/bash
cd "$(dirname "$0")"
$(ls .venv*sys/bin/python | head -1) app.py
