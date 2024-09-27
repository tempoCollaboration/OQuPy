OQuPy: Open Quantum Systems in Python
=====================================

**A Python package to efficiently simulate non-Markovian open quantum systems with process tensors.**

.. image:: https://img.shields.io/badge/arXiv-2406.16650-red
   :alt: ArXiv
   :target: https://arxiv.org/abs/2406.16650
.. image:: https://img.shields.io/badge/GitHub-tempoCollaboration%2FOQuPy-green
   :alt: GitHub
   :target: https://github.com/tempoCollaboration/OQuPy
.. image:: https://www.zenodo.org/badge/244404030.svg
   :alt: DOI
   :target: https://www.zenodo.org/badge/latestdoi/244404030

This open source project aims to facilitate versatile numerical tools to
efficiently compute the dynamics of quantum systems that are possibly
strongly coupled to structured environments. It offers the
convenient application of several numerical methods that combine the
conceptional advantages of the process tensor framework [1], with the
numerical efficiency of tensor networks.

OQuPy includes numerically exact methods (i.e. employing only
numerically well controlled approximations) for the non-Markovian
dynamics and multi-time correlations of ...

- quantum systems coupled to a single environment [2-4],
- quantum systems coupled to multiple environments [5],
- interacting chains of non-Markovian open quantum systems [6], and
- ensembles of open many-body systems with many-to-one coupling [7].

Furthermore, OQuPy implements methods to ...

- optimize control protocols for non-Markovian open quantum systems [8,9],
- compute the dynamics of an non-Markovian environment [10], and
- obtain the thermal state of a strongly coupled quantum system [11].

.. figure:: ./graphics/overview.png
   :alt: OQuPy - overview

-  **[1]** Pollock et al., `Phys. Rev. A 97,
   012127 <https://doi.org/10.1103/PhysRevA.97.012127>`__ (2018).
-  **[2]** Strathearn et al., `New J. Phys. 19(9),
   p.093009 <https://doi.org/10.1088/1367-2630/aa8744>`__ (2017).
-  **[3]** Strathearn et al., `Nat. Commun. 9,
   3322 <https://doi.org/10.1038/s41467-018-05617-3>`__ (2018).
-  **[4]** Jørgensen and Pollock, `Phys. Rev. Lett. 123,
   240602 <https://doi.org/10.1103/PhysRevLett.123.240602>`__ (2019).
-  **[5]** Gribben et al., `PRX Quantum 3,
   10321 <https://doi.org/10.1103/PRXQuantum.3.010321>`__ (2022).
-  **[6]** Fux et al., `Phys. Rev. Research 5,
   033078 <https://doi.org/10.1103/PhysRevResearch.5.033078>`__
   (2023).
-  **[7]** Fowler-Wright et al., `Phys. Rev. Lett. 129,
   173001 <https://doi.org/10.1103/PhysRevLett.129.173001>`__ (2022).
-  **[8]** Fux et al., `Phys. Rev. Lett. 126,
   200401 <https://doi.org/10.1103/PhysRevLett.126.200401>`__ (2021).
-  **[9]** Butler et al., `Phys. Rev. Lett. 132,
   060401 <https://doi.org/10.1103/PhysRevLett.132.060401>`__ (2024).
-  **[10]** Gribben et al., `Quantum, 6,
   847 <https://doi.org/10.22331/q-2022-10-25-847>`__ (2022).
-  **[11]** Chiu et al., `Phys. Rev. A 106,
   012204 <https://doi.org/10.1103/PhysRevA.106.012204>`__ (2022).


.. |binder-tutorial| image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/tempoCollaboration/OQuPy/main?filepath=tutorials%2Fquickstart.ipynb

+--------------------+--------------------------------------------------------+
| **Github**         | https://github.com/tempoCollaboration/OQuPy            |
+--------------------+--------------------------------------------------------+
| **Documentation**  | https://oqupy.readthedocs.io                           |
+--------------------+--------------------------------------------------------+
| **PyPI**           | https://pypi.org/project/oqypy/                        |
+--------------------+--------------------------------------------------------+
| **Tutorial**       | launch |binder-tutorial|                               |
+--------------------+--------------------------------------------------------+

.. image:: https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge
 :target: http://unitary.fund

-------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   pages/install

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   pages/tutorials/quickstart
   pages/tutorials/pt_tempo
   pages/tutorials/parameters
   pages/tutorials/n_time_correlations
   pages/tutorials/pt_gradient/pt_gradient
   pages/tutorials/bath_dynamics
   pages/tutorials/pt_tebd
   pages/tutorials/mf_tempo
   
.. toctree::
   :maxdepth: 1
   :caption: API Reference

   pages/api
   pages/modules

.. toctree::
   :maxdepth: 1
   :caption: Development

   pages/contributing
   pages/authors
   pages/how_to_cite
   pages/sharing
   pages/bibliography


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Consider sharing this project:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: graphics//oqupy-github-qr.png
    :align: center
    :width: 200
    :alt: OQuPy GitHub QR Code
