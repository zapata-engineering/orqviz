################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
extras = {
    # Development extras needed in every project, because the stylechecks depend on it.
    # If you need more dev deps, extend this list in your own setup.py.
    "dev": [
        "black~=22.3",
        "flake8~=3.9.0",
        "Flake8-pyproject>=0.9.0",
        "isort~=5.9.0",
        "mypy~=0.910",
        "pytest~=6.2.5",
        "pytest-cov>=2.12",
    ],
}
