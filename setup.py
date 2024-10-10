from setuptools import setup
#from numpy.distutils.core import setup
#from numpy.distutils.core import Extension

#ext1 = Extension(name='NemesisPy.Fortran.nemesisf',
#                 sources=['NemesisPy/Fortran/mulscatter.f90','NemesisPy/Fortran/spectroscopy.f90','NemesisPy/Fortran/hapke.f90'],
#                 f2py_options=['--quiet'],
#                 )

setup(name='archnemesis',
      version='1.0.0',
      description='Python implementation of the NEMESIS radiative transfer code',
      packages=['archnemesis'],
      install_requires=['numpy','matplotlib','sympy','numba','scipy','pymultinest'],
#      ext_modules=[ext1],
      )
