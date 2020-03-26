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


This should do it! (If not check if this is a known issue on github and file
one if it isn't. Thanks!)


Test Installation
-----------------

Open a interactive python3 session and type:

.. todo::

  Replace the tempo.say_hi() stuff with ``tempo.__version__`` or such like.

.. code-block:: python

  >>> import time_evolving_mps as tempo
  >>> tempo.say_hi()

This should give you the following message:

.. code-block:: none

  Hi there!
  This is TimeEvolvingMPO version 0.0.1-1 speaking!


Uninstall
---------

Uninstall TimeEvolvingMPO with pip:

.. code-block:: none

 $ python3 -m pip uninstall time_evolving_mpo
