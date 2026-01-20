from setuptools import setup, find_packages

setup(
    name="paphla",
    version="1.0.0",
    author="dongje.moon@kaist.ac.kr",
    description="PAPHLA",
    packages=find_packages(exclude=["models", "notebooks", "config"])
)