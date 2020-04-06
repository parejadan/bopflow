import setuptools


with open("requirements/base.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(install_requires=requirements)
