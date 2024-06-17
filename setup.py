import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
exec(open('ctar/version.py').read())

setuptools.setup(
    name="ctar",
    version=__version__,
    author="Ana Prieto, Martin Jinye Zhang",
    author_email="asprieto@andrew.cmu.edu, martinjzhang@gmail.com",
    description="Cell type-specific ATAC-RNA links",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/martinjzhang/cts_peak_gene",
    project_urls={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["ctar"],
    python_requires=">=3.5",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "scipy>=1.5.0",
        "scanpy>=1.6.0",
        "anndata>=0.7",
        "statsmodels>=0.11.0",
        "muon>=0.1.5",
        "pybedtools>=0.9.1",
        "biomart>=0.9.2",
#         "tqdm",
#         "fire>=0.4.0",
#         "pytest>=6.2.0",
    ],
    scripts=[
#         "bin/scdrs",
    ],
#     package_data={'scdrs': ['data/*']},
)
