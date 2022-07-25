#!/usr/bin/env python
################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################

import site
import sys

from pygit2 import Repository
from setuptools import find_namespace_packages, setup

try:
    from subtrees.z_quantum_actions.setup_extras import extras
except ImportError:
    print("Unable to import extras", file=sys.stderr)
    extras = {"dev": []}

# Workaound for https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

# This next stanza will make sure that if we're working on the @dev branch of
# this repo, that we use the @dev branch fo orquestra-sdk.
branch_name = Repository(".").head.shorthand
orquestra_sdk_dep = "orquestra-sdk @ git+ssh://git@github.com/zapatacomputing/orquestra-sdk.git"  # noqa: E501
if "feature" in branch_name:
    orquestra_sdk_dep += "@dev"

setup(
    name="orquestra-bogus",
    description="Monitoring Orquestra workflows.",
    package_dir={"": "src/python"},
    packages=find_namespace_packages(include=["orquestra.*"], where="src/python"),
    include_package_data=True,
    license="LICENSE",
    install_requires=[
        "pydantic",
        "ray[default]",
        orquestra_sdk_dep,
    ],
    extras_require=extras,
    # Without this, users of this library would get mypy errors. See also:
    # https://github.com/python/mypy/issues/7508#issuecomment-531965557
    zip_safe=False,
)
