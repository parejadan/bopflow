import setuptools


with open("requirements/base.txt") as f:
    requirements = f.read().splitlines()

with open("requirements/training.txt") as f:
    training_require = f.read().splitlines()

setuptools.setup(
    install_requires=requirements,
    extras_require={"training": training_require},
)
