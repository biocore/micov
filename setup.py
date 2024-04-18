"""micov: microbiome coverage."""
import versioneer
from setuptools import setup, find_packages


setup(name='micov',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      license='BSD-3-Clause',
      author='Daniel McDonald',
      author_email='damcdonald@ucsd.edu',
      packages=find_packages(),
      install_requires=[
          'polars',
          'matplotlib',
          'scipy',
          'click',
          'numba'],
      entry_points='''
          [console_scripts]
          micov=micov.cli:cli
      ''')
