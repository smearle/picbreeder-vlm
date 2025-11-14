from setuptools import setup, find_packages

setup(
    name="picbreeder-vlm",
    version="0.1.0",
    description="Expose rendering.py as a top-level module",
    packages=find_packages(include=["picture2d", "tools"]),
    py_modules=["rendering"],
)