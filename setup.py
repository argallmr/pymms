
from setuptools import setup

setup(
    name = "MMS Data Extractor",
    version = "0.0.1",
    author = "Matthew Argall",
    author_email = "",
    description = "Exports data from the pymms project",
    license = "?",
    keywords = "space, MMS, data",
    packages = ["pymms"],
    #long_description = read("README.md"),
    install_requires = ["numpy >= 1.8",
        "pandas", "matplotlib", "spacepy", "tqdm", "gtk"]
    )
