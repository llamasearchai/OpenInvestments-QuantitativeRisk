.PHONY: help venv install-smoke smoke clean-venv

PYTHON ?= python3
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

help:
	@echo "Targets:"
	@echo "  venv           - Create virtual environment (.venv)"
	@echo "  install-smoke  - Install minimal smoke dependencies"
	@echo "  smoke          - Run smoke tests (test_installation.py)"
	@echo "  clean-venv     - Remove .venv"

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)

venv: $(VENV)/bin/python
	@echo "Virtual environment ready at $(VENV)"

install-smoke: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-smoke.txt

smoke: venv
	$(PY) test_installation.py

clean-venv:
	rm -rf $(VENV)

