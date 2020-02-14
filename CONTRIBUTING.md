# How to Contribute
The main code structure is currently under development and therefore the
**project is not yet ready for code contributions.** If you have any question,
suggestion or general thoughts on this library we'd very much appreciate if you
share them with us. For this please feel free to file an issue in the
[Issues](https://github.com/gefux/TimeEvolvingMPO/issues) section on github.
Once we are off the ground with the code your code contributions are wellcome
as well.

For the coding bit of the project we try to follow these general guidlines:

* [the zen of python](https://www.python.org/dev/peps/pep-0020/)
* [structure](https://docs.python-guide.org/writing/structure/)
* [code style](https://www.python.org/dev/peps/pep-0008/)

The current setup uses:

* [pytest](https://docs.pytest.org) ... for testing.
* [circleCI](https://circleci.com/) ... for continuous integration.
* [pylint](https://www.pylint.org/) ... for code style checks.
* [sphinx](https://www.sphinx-doc.org) ... for generating the documentation.


## How to contribute to the code or documentation
Please use the [Issues](https://github.com/gefux/TimeEvolvingMPO/issues) and
[Pull requests](https://github.com/gefux/TimeEvolvingMPO/pulls) system on github.
If you haven't done this sort of thing before, this link
(https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github)
is a nice and compact tutorial on how to contribute to OpenSource projects
with Pull requests.

Get set up:
```bash
$ # if you havent installed python3 virtual environment already:
$ sudo apt-get install python3-venv

$ cd [the TimeEvolvingMPO repo]

$ # create a fresh virtual environment to make sure you are using the same
$ # dependencies
$ python3 -m venv venv              
$ source venv/bin/activate

$ # install the CI (contineous integration) dependencies to that virtual
$ # environment
$ python3 -m pip install -r requirements_ci.txt
$ deactivate
```

Before you make any pull request check how things are:
```bash
$ # activate the virtual environment you set up before
$ source venv/bin/activate

$ # run the tests and check if everything passed
$ python3 -m pytest --cov-report term --cov=time_evolving_mpo tests/

$ # check that the codestyle is consistant with PEP 08
$ python3 -m pylint time_evolving_mpo/

$ # check that creating the doc files still works.
$ cd ./doc && make clean && make html && cd ..
$ firefox ./doc/_build/html/index.html &
$ deactivate
```
