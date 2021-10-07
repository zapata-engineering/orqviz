import os

import setuptools

dev_requires = [
    "pytest>=3.7.1",
]

extras_require = {
    "dev": dev_requires,
}
setuptools.setup(
    name="orqviz",
    version="0.0.1",
    author="Zapata Computing, Inc.",
    author_email="info@zapatacomputing.com",
    description="""Python package for visualizing loss landscapes" \
        " of Parameterized Quantum Circuits""",
    url="https://github.com/zapatacomputing/orqviz",
    packages=setuptools.find_namespace_packages(include=["orqviz.*"], where="src"),
    package_dir={"": "src"},
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        "numpy",
        "sklearn",
        "matplotlib",
    ],
    extras_require=extras_require,
)
