OQuPy - Open Quantum Systems in Python
======================================

**A Python 3 package to efficiently compute non-Markovian open quantum
systems.**

.. image:: https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge
 :target: http://unitary.fund


This open source project aims to facilitate versatile numerical tools to
efficiently compute the dynamics of quantum systems that are possibly strongly
coupled to structured environments. It allows to conveniently apply several
numerical methods related to the time evolving matrix product operator
(TEMPO) [1-2] and the process tensor (PT) approach to open quantum
systems [3-5]. This includes methods to compute ...

- the dynamics of a quantum system strongly coupled to a bosonic environment [1-2].
- the process tensor of a quantum system strongly coupled to a bosonic environment [3-4].
- optimal control procedures for non-Markovian open quantum systems [5].
- the dynamics of a strongly coupled bosonic environment [6].
- the dynamics of a quantum system coupled to multiple non-Markovian environments [7].
- the dynamics of a chain of non-Markovian open quantum systems [8].
- the dynamics of an open many-body system with one-to-all light-matter
  coupling [9].

Up to versions 0.1.x this package was called *TimeEvolvingMPO*.

.. figure:: graphics/overview.png
    :align: center
    :alt: OQuPy - overview

- **[1]** Strathearn et al.,
  `New J. Phys. 19(9), p.093009 <http://dx.doi.org/10.1088/1367-2630/aa8744>`_
  (2017).
- **[2]** Strathearn et al.,
  `Nat. Commun. 9, 3322 <https://doi.org/10.1038/s41467-018-05617-3>`_
  (2018).
- **[3]** Pollock et al.,
  `Phys. Rev. A 97, 012127 <http://dx.doi.org/10.1103/PhysRevA.97.012127>`_
  (2018).
- **[4]** Jørgensen and Pollock,
  `Phys. Rev. Lett. 123, 240602 <http://dx.doi.org/10.1103/PhysRevLett.123.240602>`_
  (2019).
- **[5]** Fux et al.,
  `Phys. Rev. Lett. 126, 200401 <https://link.aps.org/doi/10.1103/PhysRevLett.126.200401>`_
  (2021).
- **[6]** Gribben et al.,
  `arXiv:2106.04212 <http://arxiv.org/abs/2106.04212>`_
  (2021).
- **[7]** Gribben et al.,
  `PRX Quantum 3, 10321 <https://link.aps.org/doi/10.1103/PRXQuantum.3.010321>`_
  (2022).
- **[8]** Fux et al.,
  `arXiv:2201.05529 <http://arxiv.org/abs/2201.05529>`_ (2022).
- **[9]** Fowler-Wright at al.,
  `Phys. Rev. Lett. 129, 173001 <https://doi.org/10.1103/PhysRevLett.129.173001>`_
  (2022).

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
   pages/bibliography


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
