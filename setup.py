# Always prefer setuptools over distutils
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

extensions = [Extension("mrina.solver_omp", ["mrina/solver_omp.pyx"])]

# This call to setup() does all the work
setup(
    name="mrina",
    version="0.2.7",
    description="Library for MRI noise analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://mrina.readthedocs.io/",
    author="Lauren Partin, Daniele Schiavazzi, Carlos Sing-Long",
    author_email="example@email.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent"
    ],
    packages=["mrina"],
    include_package_data=True,
    install_requires=["numpy"],
    ext_modules = cythonize(extensions,
                            annotate=True,
                            compiler_directives={'language_level':3}
                            ),
    include_dirs=[np.get_include()]
)