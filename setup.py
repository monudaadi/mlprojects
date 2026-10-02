from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    '''This function will return the list of requirements'''
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name="mlproject",
    version="0.1.0",
    author="Monu",
    author_email="daadimonu@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)