# How to Contribute
The main code structure is currently under development and therefore the
**project is not yet ready for code contributions.** If you have any question,
suggestion or general thoughts on this library we'd very much appreciate if you
share them with us. For this please feel free to file an issue in the
[Issues](https://github.com/tempoCollaboration/TimeEvolvingMPO/issues) section
on github. Once we are off the ground with the code your code contributions are
very welcome as well.

For the coding bit of the project we try to follow these general guidlines:

* [the zen of python](https://www.python.org/dev/peps/pep-0020/)
* [structure](https://docs.python-guide.org/writing/structure/)
* [code style](https://www.python.org/dev/peps/pep-0008/)

The current setup uses:

* [pytest](https://docs.pytest.org) ... for testing.
* [pylint](https://www.pylint.org/) ... for code style checks.
* [sphinx](https://www.sphinx-doc.org) ... for generating the documentation.
* [tox](https://tox.readthedocs.io) ... for testing with different environments.
* [travis](https://travis-ci.com) ... for continues integration.

## How to contribute to the code or documentation
Please use the
[Issues](https://github.com/tempoCollaboration/TimeEvolvingMPO/issues) and
[Pull requests](https://github.com/tempoCollaboration/TimeEvolvingMPO/pulls)
system on github. If you haven't done this sort of thing before, this link
(https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github)
is a nice and compact tutorial on how to contribute to OpenSource projects
with Pull requests.

### Get set up
#### Git
ToDo: Describe the git & pull request business

```bash
$ # make sure you have python3.6 installed
$ sudo apt-get install python3.6
$ # make sure you have tox installed
$ python3 -m pip install tox
```

### Test all
Before you make any pull request check how things are by navigating to the
repositories directory and simply typing:
```bash
$ tox
```
This performs a bunch of tests. If they all pass it will finish with something
like:
```bash
  py36: commands succeeded
  style: commands succeeded
  docs: commands succeeded
  congratulations :)
```

#### pytest
You can run the pytests on python3.6 with:
```bash
$ tox -e py36
```
and you can pick a specific test to run with:
```bash
$ tox -e py36 say_hi_test.py
```

#### pylint
This checks the code [code style](https://www.python.org/dev/peps/pep-0008/)
with pylint:
```bash
$ tox -e style
```
and you can check the style of a specific file with:
```bash
$ tox -e style say_hi.py
```

#### sphinx
This invoces a sphinx-build to build the HTML documentation
```bash
$ tox -e docs
```
You can view the generated documentation by opening it in a browser:
```bash
$ firefox ./docs/_build/html/index.html &
```
