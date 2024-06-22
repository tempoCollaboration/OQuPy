.. _install-label:

Installation
============


How To Install
--------------

* Make sure you have `python3.10` or higher installed

  .. code-block:: none

    $ python3 --version
    Python 3.10.14


* Make sure you have `pip3` version 23.0 or higher installed

  .. code-block:: none

    $ python3 -m pip --version
    pip 23.3.1 from /home/gefux/anaconda3/envs/oqupy-ci/lib/python3.10/site-packages/pip (python 3.10)


* Install OQuPy via pip

  .. code-block:: none

    $ python3 -m pip install oqupy


Test Installation
-----------------

Open a interactive python3 session and type:

.. code-block:: python

  >>> import oqupy
  >>> oqupy.__version__

This should give you the following message:

.. code-block:: none

  '0.4.0'


Uninstall
---------

Uninstall OQuPy with pip:

.. code-block:: none

 $ python3 -m pip uninstall oqupy
