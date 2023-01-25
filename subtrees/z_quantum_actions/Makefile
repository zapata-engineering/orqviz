################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
TOP_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(TOP_DIR)/variables.mk

# This target will list the possible valid targets available.
default:
	@echo --------------------------------------------
	@echo '=> No target chosen: Choose from the following:'
	@echo
	@grep -E '^\w+(\-default)?:' $(TOP_DIR)/$(firstword $(MAKEFILE_LIST)) \
	       | sed -r 's/-default//g; /default/d ; s/(.*)/\t make \1/g ; s/:.*$$//g'

export VENV_NAME := my_little_venv
ifeq ($(OS), Windows_NT)
  VENV_BINDIR := Scripts
  PYTHON_EXE := python.exe
  # A workaround for Windows and Bash:
  # By default, PATH will give the Windows Path, but we need to get the Path for the shell we'll be using
  SHELL_PATH := $(shell env echo $$PATH)
  PYTHON := $(shell env PATH="${VENV_NAME}/${VENV_BINDIR}:${SHELL_PATH}" ${PYTHON_EXE} -c 'import sys; print(sys.executable)')
  # Convert a Windows path to a Unix path for Windows Github Actions
  PYTHON := /$(subst \,/,$(subst :\,/,$(PYTHON)))
else
  VENV_BINDIR := bin
  PYTHON_EXE := python3
  PYTHON := $(shell env PATH="${VENV_NAME}/${VENV_BINDIR}:${PATH}" ${PYTHON_EXE} -c 'import sys; print(sys.executable)')
endif

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
	${PYTHON_EXE} -m venv ${VENV_NAME}
	"${VENV_NAME}/${VENV_BINDIR}/${PYTHON_EXE}" -m pip install --upgrade pip
	"${VENV_NAME}/${VENV_BINDIR}/${PYTHON_EXE}" -m pip install -e '.[dev]'

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


# Option explanation:
# - '--cov=src' - turn on measuring code coverage. It outputs the results in a
#    '.coverage' binary file. It's passed to other commands like
#    'python -m coverage report'
coverage-default:
	$(PYTHON) -m pytest \
		--cov=src \
		--cov-fail-under=$(MIN_COVERAGE) tests \
		--no-cov-on-fail \
		&& echo Code coverage Passed the $(MIN_COVERAGE)% mark!


# Reads the code coverage stats from '.coverage' file and prints a textual,
# human-readable report to stdout.
show-coverage-text-report-default:
	$(PYTHON) -m coverage report --show-missing


style-default: flake8p mypy black isort
	@echo This project passes style!

muster-default: style coverage
	@echo This project passes muster!

build-system-deps-default:
	:

# Gets the next version of an installed package
# Note: on CI we only run this step, this means we use the github_actions target as a dependency.
# Because of this, we don't update the $(PYTHON) variable and have to manually build the path to our venv Python.
get-next-version-default: github_actions
	"${VENV_NAME}/${VENV_BINDIR}/${PYTHON_EXE}" subtrees/z_quantum_actions/bin/get_next_version.py $(PACKAGE_NAME)

# This is what converts the -default targets into base target names.
# Do not remove!!!
%: %-default
	@true
