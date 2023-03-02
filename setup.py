#!/usr/bin/env python3

import subprocess
from pathlib import Path
from typing import List

from setuptools import setup


def get_cmd(cmd: List[str]) -> str:
    try:
        return (
            subprocess.check_output(cmd, cwd=Path(__file__).parent)
            .decode("ascii")
            .strip()
        )
    except Exception:
        return ""


def get_sha() -> str:
    """Get git short"""
    return get_cmd(["git", "rev-parse", "--short", "head"])


def get_tag() -> str:
    """Get last git tag"""
    return get_cmd(["git", "describe", "--tags", "--exact"])


def main():
    if (version := get_tag()) == "":
        version = f"git+{get_sha()}"
    assert version != "", "Unable to get tag or sha"
    setup(version=version)


if __name__ == "__main__":
    main()
