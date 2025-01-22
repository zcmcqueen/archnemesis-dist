from setuptools import setup

setup(name='archnemesis',
      version='1.0.0',
      description='Python implementation of the NEMESIS radiative transfer code',
      packages=['archnemesis'],
      install_requires=['numpy','matplotlib','numba','scipy','pymultinest','cdsapi','pygrib','joblib','h5py','basemap'],
      )
