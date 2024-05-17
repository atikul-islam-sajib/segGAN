import os
from setuptools import setup, find_namespace_packages
from setuptools import find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="segGAN",
    version="0.0.1",
    author="Atikul Islam Sajib",
    author_email="atikulislamsajib137@gmail.com",
    description=("A part of the segGAN package"),
    license="MIT",
    keywords="example documentation tutorial",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    long_description=read("README.md"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: MIT Approved :: MIT License",
    ],
)
