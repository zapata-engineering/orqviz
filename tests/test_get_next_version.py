################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import re
import subprocess

import pytest

from bin import get_next_version


@pytest.mark.parametrize(
    "version_str",
    [
        # right after release
        "1.2.3",
        # a few commits after the release
        "0.1.1.dev73+g178469c.d20211213",
        # no release tag yet
        "0.1.dev82+gf24acc4",
    ],
)
def test_parse_version_str(version_str):
    semver = get_next_version.parse_version_str(version_str)
    assert semver.version_str == version_str


@pytest.mark.parametrize(
    "version_str,bumped",
    [
        # right after release
        ("1.2.3", "1.3.0"),
        # a few commits after the release
        ("0.1.1.dev73+g178469c.d20211213", "0.2.0"),
        # no release tag yet
        ("0.1.dev82+gf24acc4", "0.2.0"),
    ],
)
def test_bumping_version(version_str, bumped):
    semver = get_next_version.parse_version_str(version_str)
    assert semver.bump_minor.version_str == bumped
