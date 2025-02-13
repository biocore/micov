"""micov: microbiome coverage."""

import versioneer
from setuptools import find_packages, setup

setup(
    name="micov",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="BSD-3-Clause",
    author="Daniel McDonald",
    author_email="damcdonald@ucsd.edu",
    packages=find_packages(),
    install_requires=[
        "polars-u64-idx >= 1.21",
        "matplotlib >= 3.9",
        "duckdb",
        "scipy",
        "click",
        "numba",
        "pyarrow<16.0.0",
    ],
    entry_points="""
          [console_scripts]
          micov=micov.cli:cli
      """,
)
