archNEMESIS
===========

.. image:: https://img.shields.io/badge/readthedocs-latest-blue
   :target: https://archnemesis.readthedocs.io

.. image:: https://img.shields.io/badge/github-code-green
   :target: https://github.com/juanaldayparejo/archnemesis-dist

.. image:: https://img.shields.io/badge/NEMESIS-reference-yellow
   :target: https://doi.org/10.1016/j.jqsrt.2007.11.006


__________

This website includes the documentation regarding the Python version of the NEMESIS (Non-linear Optimal Estimator for MultivariatE
Spectral analySIS) planetary atmosphere radiative transfer and retrieval code. 

The main description of the NEMESIS code was published by `Irwin et al. (2008) <https://doi.org/10.1016/j.jqsrt.2007.11.006>`_.
The original Fortran version of the code is `available here <https://doi.org/10.5281/zenodo.4303976>`_.

In this website, we aim to provide a more practical description of the code, including retrieval examples applied to different observing geometries or physical parameterisations.

**NOTE:** At this stage, documentation is under development.

Install archNEMESIS
--------------------

The latest version of code has to be downloaded from `Github <https://github.com/juanaldayparejo/archnemesis-dist.git>`_.

Once the code has been downloaded from Github, move the archnemesis-dist/ package to a desired path. Then, inside the package, type ::

$ pip install --editable .

This will install the NemesisPy package, but with the ability to update any changes made to the code (e.g., when introducing new model parameterisations).

In the future, we aim to release official versions to The Python Package Index (PyPI), so that it can be directly installed using pip.


Revision history
-----------------------------

- 1.0.0 (1 August, 2024)
    - First version of the code.

Dependencies on other Python packages
-----------------------------

- `numpy <https://numpy.org/>`_: Used widely throughout the code to define N-dimensional arrays and perform mathematical operations (e.g., matrix multiplication).
- `matplotlib <https://matplotlib.org/>`_: Used to create visualizations. 
- `numba <https://numba.pydata.org/>`_: Used in specific functions to include the JIT compiler decorator and speed up the radiative transfer calculations.

.. toctree::
   :maxdepth: 2

.. toctree::
   :caption: General Structure
   :hidden:
   
   documentation/general_structure.ipynb
 
.. toctree::
   :caption: Reference classes
   :hidden:
   
   documentation/reference_classes.ipynb

.. toctree::
   :caption: Model parameterisations
   :hidden:
   
   documentation/model_parameterisations.ipynb

.. toctree::
   :caption: Forward Model
   :hidden:
   
   documentation/forward_model.ipynb
 
.. toctree::
   :caption: Retrievals
   :hidden:
   
   documentation/retrievals.ipynb
   
.. toctree::
   :caption: Examples
   :hidden:
   
   examples


