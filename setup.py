import setuptools


setuptools.setup(
    install_requires=["tensorflow==2.3.1"],
    extras_require={"training": ["opencv-python==4.1.1.26"]}
)
