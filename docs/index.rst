TimeEvolvingMPO reference documentation
=======================================

**A Python 3 package to efficiently compute non-Markovian open quantum
systems.**

.. image:: https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge
 :target: http://unitary.fund

This open source project aims to facilitate versatile numerical tools to
efficiently compute the dynamics of quantum systems that are possibly strongly
coupled to structured environments. It allows to conveniently apply the so
called time evolving matrix product operator method (TEMPO) [1], as well as the
process tensor TEMPO method (PT-TEMPO) [2]. An extension to the methods
described in [3] and [4] are work in progress and supported by the
`unitary fund <http://unitary.fund>`_.

- **[1]** Strathearn et al.,  *Efficient non-Markovian quantum dynamics using
  time-evolving matrix product operators*, Nat. Commun. 9, 3322 (2018).
- **[2]** Fux et al., *Efficient exploration of Hamiltonian parameter space for
  optimal control of non-Markovian open quantum systems*,
  Phys. Rev. Lett. 126, 200401(2021).
- **[3]** Gribben et al., *Using the Environment to Understand non-Markovian
  Open Quantum Systems*, arXiv:2106.04212 (2021).
- **[4]** Gribben et al., *Exact dynamics of non-additive environments in
  non-Markovian open quantum systems*, arXiv:2109.08442 (2021).

.. |binder-tutorial| image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/tempoCollaboration/TimeEvolvingMPO/master?filepath=tutorials%2Ftutorial_01_quickstart.ipynb

+--------------------+---------------------------------------------------------------------------------------------------------------+
| **Github**         | https://github.com/tempoCollaboration/TimeEvolvingMPO                                                         |
+--------------------+---------------------------------------------------------------------------------------------------------------+
| **Documentation**  | https://TimeEvolvingMPO.readthedocs.io                                                                        |
+--------------------+---------------------------------------------------------------------------------------------------------------+
| **PyPI**           | https://pypi.org/project/time-evolving-mpo/                                                                   |
+--------------------+---------------------------------------------------------------------------------------------------------------+
| **Tutorial**       | launch |binder-tutorial|                                                                                      |
+--------------------+---------------------------------------------------------------------------------------------------------------+

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   pages/install

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   pages/tutorial_01_quickstart/tutorial_01_quickstart
   pages/tutorial_02_pt_tempo/tutorial_02_pt_tempo

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
