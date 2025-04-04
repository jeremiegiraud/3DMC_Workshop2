from setuptools import setup, find_packages

setup(
    name='GeoMos',
    version='0.1.0',
    url='https://github.com/jeremiegiraud/GeoMos_nullspace',
    description='setup for null space shuttles code',
    packages=find_packages(),
    install_requires=[
        # Github Repository.
        'tomofasttools @ git+https://git@github.com/TOMOFAST/Tomofast-tools.git',
        'numpy>=1.23.3',
        'matplotlib>=3.5.3',
        'colorcet>=3.0.1',
        'scipy>=1.10.1',
        'scikit-fmm>=2022.8.15',
    ]
)
