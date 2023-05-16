import setuptools
import sys

#Long description
with open("Readme.rst", "r") as fh:
    long_description = fh.read()

#Read requeriments
with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

# We raise an error if trying to install with python2
if sys.version[0] == '2':
    print("Error: This package must be installed with python3")
    sys.exit(1)

#Package configuration
setuptools.setup(
    name="sizingmicrogrids",
    version="1",
    author="Sebastian Castellanos Buitrago",
    author_email="sebastian.castellanos@udea.edu.co",
    description="Optimization and simulation tool for sizing microgrid",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/SENECA-UDEA/microgrids_sizing",
    packages=setuptools.find_packages(install_requires),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    #Install requeriments
    install_requires=install_requires,
    include_package_data=True,
)

