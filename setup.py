from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
      name = "nasa-pymms",
      version = "0.1.0",
      author = "Matthew R. Argall",
      author_email = "argallmr@gmail.com",
      description = "Access data from the MMS mission via its API.",
      long_description = long_description,
      long_description_content_type = "text/markdown",
      url = "https://github.com/argallmr/pymms",
      license = "MIT",
      classifiers = ["Programming Language :: Python :: 3",
                     "Operating System :: OS Independent",
                     "License :: OSI Approved :: MIT License",
                     "Development Status :: 4 - Beta",
                     "Topic :: Scientific/Engineering :: Physics",
                     "Topic :: Scientific/Engineering :: Astronomy",
                     "Intended Audience :: Science/Research",
                     "Natural Language :: English"],
      keywords = "physics space-physics MMS",
      packages = ["pymms"],
      install_requires = ["numpy >= 1.8",
                          "requests",
                          "scipy", 
                          "tqdm",
                          "cdflib",
                          "matplotlib"
                          ],
      python_requires = '>=3.6'
      )
