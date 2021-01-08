.. _install-label:

Installation
============


How To Install
--------------

* Make sure you have `python3.6` or higher installed

  .. code-block:: none

    $ python3 --version
    Python 3.6.9


* Make sure you have `pip3` version 20.0 or higher installed

  .. code-block:: none

    $ python3 -m pip --version
    pip 20.0.2 from /home/gefux/.local/lib/python3.6/site-packages/pip (python 3.6)


* Install TimeEvolvingMPO via pip

  .. code-block:: none

    $ python3 -m pip install time_evolving_mpo


Test Installation
-----------------

Open a interactive python3 session and type:

.. code-block:: python

  >>> import time_evolving_mpo as tempo
  >>> tempo.say_hi()

This should give you the following message:

.. code-block:: none

  '0.0.1-2'


Uninstall
---------

Uninstall TimeEvolvingMPO with pip:

.. code-block:: none

 $ python3 -m pip uninstall time_evolving_mpo
