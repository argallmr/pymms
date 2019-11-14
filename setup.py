from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "MMS-SDC-API",
    version = "0.0.1",
    author = "Matthew R. Argall",
    author_email = "argallmr@gmail.com",
    description = "Access data from the MMS mission via its API.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/argallmr/pymms"
    license = "MIT",
    keywords = "space, MMS, data",
    packages = ["pymms"],
    install_requires = ["numpy >= 1.8",
                        "pandas", 
                        "matplotlib", 
                        "requests",
                        "spacepy", 
                        "tqdm"
    ]
    python_requires = '>=3.6',
)
