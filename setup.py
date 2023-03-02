#!/usr/bin/env python3

import subprocess
from pathlib import Path
from typing import List

from setuptools import setup


def get_cmd(cmd: List[str]) -> str:
    return (
        subprocess.check_output(cmd, cwd=Path(__file__).parent).decode("ascii").strip()
    )


def get_sha() -> str:
    """Get current git commit short"""
    return get_cmd(["git", "rev-parse", "--short", "HEAD"])


def get_last_tag() -> str:
    """Get the last most recent git tag"""
    return get_cmd(["git", "describe", "--tags", "--abbrev=0"])


def get_tag() -> str:
    """Get the current git tag, will throw exception if it isn't tagged"""
    return get_cmd(["git", "describe", "--tags", "--exact"])


def main():
    try:
        version = get_tag()
    except subprocess.SubprocessError:
        # if current commit
        version = f"{get_last_tag()}+git{get_sha()}"
    setup(version=version)


if __name__ == "__main__":
    main()
