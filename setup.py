from setuptools import setup, find_packages

version = '0.0.1'

setup(name='micov',
      version=version,
      license='BSD-3-Clause',
      author='Daniel McDonald',
      author_email='damcdonald@ucsd.edu',
      packages=find_packages(),
      entry_points='''
          [console_scripts]
          micov=micov.cli:cli
      ''')
