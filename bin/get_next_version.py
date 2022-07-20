#! /usr/bin/env python3
################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################

# This script ia intended to be run from within a Github Action.
# Reads current project version, bumps "minor", and sets the "next_version" output
# variable using Github's special stdout syntax.

import sys
import re
import typing as t
from importlib.metadata import version


class Semver(t.NamedTuple):
    major: int
    minor: int
    patch: t.Optional[int]
    pre: t.Optional[str]

    @property
    def version_str(self):
        version = f"{self.major}.{self.minor}"
        if self.patch is not None:
            version += f".{self.patch}"
        if self.pre:
            version += f".{self.pre}"

        return version

    @property
    def bump_minor(self):
        return Semver(
            major=self.major,
            minor=self.minor + 1,
            patch=0,
            pre=None,
        )


SEMVER_REGEX = (
    r"(?P<major>[0-9]+)\.(?P<minor>[0-9]+)(\.(?P<patch>[0-9]+))?([-\.](?P<pre>.+))?"
)


def parse_version_str(version: str) -> Semver:
    # TODO: make it more resilient to versions with missing components if we need it.
    # We should be fine for some time, because setuptools_scm return all components.
    match = re.match(SEMVER_REGEX, version)
    if match is None:
        raise ValueError(f"Can't parse version string '{version}'")

    groups = match.groupdict()

    # re sets up `None` as the value for a missing group
    if groups["patch"] is not None:
        patch = int(groups["patch"])
    else:
        patch = None

    return Semver(
        major=int(groups["major"]),
        minor=int(groups["minor"]),
        patch=patch,
        pre=groups.get("pre"),
    )


def _set_github_output(name: str, value):
    # Special Github syntax for setting outputs from steps in Github Actions. See:
    # https://docs.github.com/en/actions/reference/workflow-commands-for-github-actions#setting-an-output-parameter
    print(f"::set-output name={name}::{value}")


def main(package_name):
    """Run the actual script logic.

    Args:
        version_verride: should contain Github Action input. Empty strings are treated
            as "nil" values. This is because that's how bash passes nils.
    """

    current_version = version(package_name)
    print(f"Read current version as: {current_version}")

    next_version = parse_version_str(current_version).bump_minor.version_str

    print(f"Next version: {next_version}")
    _set_github_output("next_version", next_version)


if __name__ == "__main__":
    package_name = sys.argv[1]
    main(package_name=package_name)
