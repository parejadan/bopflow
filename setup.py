import setuptools


setuptools.setup(
    install_requires=["tensorflow==2.7.2"],
    extras_require={"training": ["opencv-python==4.1.1.26"]}
)
