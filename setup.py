from setuptools import setup

setup(
    name="wdm",
    py_modules=["wdm"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
