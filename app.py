#!/usr/bin/env python3
"""Точка входа для CLI.

Использование:
    python app path/to/hh.csv

Создаёт x_data.npy and y_data.npy рядом с входным CSV.
"""

from hh_preprocess.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
