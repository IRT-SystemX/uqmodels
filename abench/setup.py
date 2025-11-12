from setuptools import setup, find_packages

setup(
    name="abench",
    version="1.0.5",
    description="Agnostic benchmark",
    author="Kevin PASINI",
    author_email="Kevin.pasini@irt-systemX.fr",
    packages=["abench",
              "abench.benchmark",
              "abench.component",
              "abench.data_loader",
              "abench.data_loader.timeseries",
              "abench.data_loader.image",
              "abench.metric",
              "abench.store",
              "abench.visu"],
    install_requires=[
        "pillow",
        "matplotlib>=3.4.3",
        "matplotlib-inline>=0.1.2",
        "numpy>=1.20.3",
        "scikit-learn>=0.24.2",
    ],
    license="",
)
