################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
TOP_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
include $(TOP_DIR)/variables.mk

# This target will list the possible valid targets available.
default:
	@echo --------------------------------------------
	@echo '=> No target chosen: Choose from the following:'
	@echo
	@grep -E '^\w+(\-default)?:' $(TOP_DIR)/$(firstword $(MAKEFILE_LIST)) \
	       | sed -r 's/-default//g; /default/d ; s/(.*)/\t make \1/g ; s/:.*$$//g'


export VENV_NAME := my_little_venv
PYTHON := $(shell PATH="${VENV_NAME}/bin:${PATH}" python3 -c 'import sys; print(sys.executable)')
REPO := $(shell git config --get remote.origin.url)
PYTHON_MOD := $(shell find src -maxdepth 3 -mindepth 3 -type d | sed '/.*cache/d; s/src\/python\/// ; s/\//./')
PACKAGE_NAME := "foo"

ifeq ($(PYTHON),)
$(error "PYTHON=$(PYTHON)")
else
$(info -------------------------------------------------------------------------------)
$(info You are using PYTHON: $(PYTHON))
$(info Python Version: $(shell $(PYTHON) --version))
$(info Repository: $(REPO))
$(info Python Modules Covered: $(PYTHON_MOD))
$(info -------------------------------------------------------------------------------)
endif

# Clean out all Pythonic cruft
clean-default:
	@find . -regex '^.*\(__pycache__\|\.py[co]\)$$' -delete;
	@find . -type d -name __pycache__ -exec rm -r {} \+
	@find . -type d -name '*.egg-info' -exec rm -rf {} +
	@find . -type d -name .mypy_cache -exec rm -r {} \+
	@rm -rf .pytest_cache;
	@rm -rf tests/.pytest_cache;
	@rm -rf dist build
	@rm -f .coverage*
	@echo Finished cleaning out pythonic cruft...

install-default: clean
	$(PYTHON) -m pip install --upgrade pip && \
		$(PYTHON) -m pip install .

dev-default: clean
	$(PYTHON) -m pip install -e .[dev]

# Why we want to use `python3` and not `$(PYTHON)` here.
# In order to enable running make commands both from CICD and locally
# we need to use virtualenv when running on GitHub Actions, as otherwise 
# we might get into rare and hard to debug edge cases (as we did in the past).
# After this action is executed $(PYTHON) will get resolved to the Python version
# from virtual environment.
# This make task is used to create new virtual environment so we want to use 
# `python3` here as it's more explicit, because $(PYTHON) would evaluate to 
# something else after executing this task, which might be confusing.
github_actions-default:
	python3 -m venv ${VENV_NAME} && \
		${VENV_NAME}/bin/python3 -m pip install --upgrade pip && \
		${VENV_NAME}/bin/python3 -m pip install -e '.[dev]'

# flake8p is a wrapper library that runs flake8 from config in pyproject.toml
# we can now use it instead of flake8 to lint the code
flake8p-default: clean
	$(PYTHON) -m flake8p --ignore=E203,E266,F401,W503 --max-line-length=88 src tests

mypy-default: clean
	@echo scanning files with mypy: Please be patient....
	$(PYTHON) -m mypy src tests

black-default: clean
	$(PYTHON) -m black --check src tests

isort-default: clean
	$(PYTHON) -m isort --check src tests

test-default:
	$(PYTHON) -m pytest tests

coverage-default:
	$(PYTHON) -m pytest \
		--cov=src \
		--cov-fail-under=$(MIN_COVERAGE) tests \
		--no-cov-on-fail \
		--cov-report xml \
		&& echo Code coverage Passed the $(MIN_COVERAGE)% mark!


style-default: flake8p mypy black isort
	@echo This project passes style!

muster-default: style coverage
	@echo This project passes muster!

build-system-deps-default:
	:

get-next-version-default: github_actions
	${VENV_NAME}/bin/python3 subtrees/z_quantum_actions/bin/get_next_version.py $(PACKAGE_NAME)

# This is what converts the -default targets into base target names.
# Do not remove!!!
%: %-default
	@true
