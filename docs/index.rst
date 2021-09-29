TimeEvolvingMPO reference documentation
=======================================

**A Python 3 package to efficiently compute non-Markovian open quantum
systems.**

This open source project aims to facilitate versatile numerical tools to
efficiently compute the dynamics of quantum systems that are possibly strongly
coupled to a structured environment. It allows to conveniently apply the so
called time evolving matrix product operator method (TEMPO) [1], as well as
the process tensor TEMPO method (PT-TEMPO) [2].

- **[1]** A. Strathearn, P. Kirton, D. Kilda, J. Keeling and
  B. W. Lovett,  *Efficient non-Markovian quantum dynamics using
  time-evolving matrix product operators*, Nat. Commun. 9, 3322 (2018).
- **[2]** G. E. Fux, E. Butler, P. R. Eastham, B. W. Lovett, and
  J. Keeling, *Efficient exploration of Hamiltonian parameter space for
  optimal control of non-Markovian open quantum systems*,
  Phys. Rev. Lett. 126, 200401(2021).


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
