from setuptools import setup,find_packages
from typing import List

def getreq(path):
    with open(path) as p:
        packages = p.readlines()
        packages = [p.replace("\n","") for p in packages if "-e ." != p]
    return packages

setup(
    name = 'generic_ml_project',
    version='0.0.1',
    description="This is generic p  roject for machine learning that can used for reference",
    author="anish",
    packages=find_packages(),
    requires= getreq("requirements.txt"),
    )


if __name__ == '__main__':
    print(getreq("requirements.txt"))