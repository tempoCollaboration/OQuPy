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
system on github.

If you are familiar with the process of contributing to an open source project
please make sure you tick all (or most) appropriate boxes in
[`PULL_REQUEST_TEMPLATE.md`](https://github.com/tempoCollaboration/TimeEvolvingMPO/blob/master/PULL_REQUEST_TEMPLATE.md)
before submitting a pull request.

If you haven't done this sort of thing before, this link
(https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github)
is a nice and compact tutorial on how to contribute to OpenSource projects
with Pull requests. Also, there is a detailed description on how to contribute
to this project below.

### Overview:

1. Discuss the issue in the [Issue section](https://github.com/tempoCollaboration/TimeEvolvingMPO/issues)
2. Create a fork of this repository on Github
3. Setup your local environment
4. Clone your fork to your local machine
5. Create a new branch pr/topic-name
6. Make your changes and tick the boxes
7. Check your code: run tests
8. Tidy up: Rebase and squash
9. Create a [pull requests](https://github.com/tempoCollaboration/TimeEvolvingMPO/pulls)
10. Make changes to your pull request


### 1. Discuss the issue in the Issue section
There are multiple ways of contributing to project, for example:

* Pointing out a bug or a typo,
* Suggesting an improvement on the API or backend,
* Suggesting new features,
* Extending the documentation or tutorials,
* Contributing bugfixes,
* Contributing code to improve the API or backend,
* or, simply sharing general thoughts on the package.

To ensure that your contribution has the best effect on the package it is vital
to discuss them first. Communication is essential to make sure your contribution
fits to the goals of the project.

Go to the
[Issue section](https://github.com/tempoCollaboration/TimeEvolvingMPO/issues)
and file an issue to share your thoughts. Ideally you discuss your plans with
others (particularly a maintainer) before you invest any significant amount of
time to avoid overlap or conflicts with other contributions. This is essential
to ensure that your work fits well into the project.


### 2. Create a fork of this repository on Github
If you have discussed your plans in the
[Issue section](https://github.com/tempoCollaboration/TimeEvolvingMPO/issues)
and would like to make a direct contribution to the code or documentation,
you will need to [create a fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo)
of the repository on your github account. This creats a linked copy on your
account which you can modify to your liking. You can then
[create a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request)
to suggest your changes to this project.


### 3. Clone your fork to your local machine
Next, you will need to
[clone](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)
to your fork to your local machine and add the original repository as a remote
resource with:

```bash
$ #if you use ssh:
$ git remote add upstream git@github.com:tempoCollaboration/TimeEvolvingMPO.git
$ #if you use https
$ git remote add upstream https://github.com/tempoCollaboration/TimeEvolvingMPO.git

$ git fetch upstream
$ git branch --set-upstream-to=upstream/master master
```

With this, if there is a change to the original repository, you can bring your
fork up to date by pulling from the `upstream` remote repository (the original
repository) and pushing to the `origin` remote repository (your fork).

```bash
$ git checkout master
$ git pull upstream master
$ git push origin master
```

### 4. Setup your local environment
As a development environment you will need `git`, `python3.6`, `pip3` and `tox`
installed.

You need `git` to
[pull, commit and push](https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control)
changes to the repository, while `tox` allows you to run tests on the package
in an virtual environment (to avoid that the result of the tests depends on the
local setup). If you are unfamiliar with `git`,
[this](https://swcarpentry.github.io/git-novice/) is a good place to start
learning about it. Currently the code is tested against `python3.6`, which makes it
neccessary to have this version of the python interpreter installed in the
development environment.

You may wish to create a virtual environment dedicated to the development
of this package with `conda` or `venv` to avoid interference with other
applications.

Finally, install the required python package dependencies with (executed in the
repositories main directory):
```
$ python3 -m pip install -r requirements_ci.txt
```

### 5. Create a new branch pr/topic-name
To group together all the changes you make, you should create a new temporary
branch. We suggest to use a name of the form `pr/topic-name` for this branch.
Create the new branch on your fork with:
```bash
$ git checkout master
$ git checkout -b pr/add-documentation-to-dynamics
$ git push --set-upstream origin pr/add-documentation-to-dynamics
```

### 6. Make your changes and tick the boxes
Now, you are all set to make your changes in
[modify-add-commit](https://swcarpentry.github.io/git-novice/04-changes/)
cycles and push them to your fork.

Later, when you create the pull request (section 9) these changes will be
reviewed by a maintainer who will check all neccessary boxes in
[`PULL_REQUEST_TEMPLATE.md`](https://github.com/tempoCollaboration/TimeEvolvingMPO/blob/master/PULL_REQUEST_TEMPLATE.md)
to guarantee the quality of the package. Therefore, during the process of
creating your changes, it is advisable to occasionally compare your work
against this list and periodically run the automated tests (see section 7).
This checklist is there to *help* all contributors to produce high quality code,
not to deter you from contributing.


### 7. Check your code: run tests
Three of the boxes in
[`PULL_REQUEST_TEMPLATE.md`](https://github.com/tempoCollaboration/TimeEvolvingMPO/blob/master/PULL_REQUEST_TEMPLATE.md) are checked automatically when new commits are
added to the project, namely:

* execute the tests that check the functionality of the package (with `pytest`)
* test the coding style (with `pylint`)
* check that the documentation can be generated (with `sphinx`)


#### 7.1 test all
You can run all three tests by simply running tox in the directory of your
local clone:
```bash
$ tox
```
This performs a bunch of tests. If they all pass, it will finish with something
like:
```bash
  py36: commands succeeded
  style: commands succeeded
  docs: commands succeeded
  congratulations :)
```

#### 7.2 test pytest only
You can run the pytests on python3.6 with:
```bash
$ tox -e py36
```
and you can pick a specific test to run with:
```bash
$ tox -e py36 say_hi_test.py
```

#### 7.3 test pylint only
This checks the code [code style](https://www.python.org/dev/peps/pep-0008/)
with pylint:
```bash
$ tox -e style
```
and you can check the style of a specific file with:
```bash
$ tox -e style say_hi.py
```

#### 7.4 test sphinx only
This invoces a sphinx-build to build the HTML documentation
```bash
$ tox -e docs
```
You can view the generated documentation by opening it in a browser:
```bash
$ firefox ./docs/_build/html/index.html &
```

### 8. Tidy up: squash and rebase
A typical development process involves a lot of bugfixes and repeated
redesigning of the code. Although it is good practice to capture every step
of the process in its own commit it can clutter the commit history make it
hard to follow. Therefore, once you think your contribution is ready for
sharing, it is probably appreciated by your peers and the maintainers if you
tidy up (squash) the commit history of your code contribution. Also, in case
that there have been other code contributions since you created the pull
request branch, rebasing the branch can help avoiding merge conflicts later.
Both, squashing and rebasing can be performed with the command
`git rebase -i ...` and are described
[here](https://git-scm.com/book/en/v2/Git-Branching-Rebasing). This command
is destructive (i.e. it is possible to lose information), and therefore you
should create a local backup before using it. However, if you are
unsure, feel free to skip this tidying up process.  


### 9. Create a pull requests
You can now
[create a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).
This request will then be reviewed by a maintainer who will check all neccessary boxes in
[`PULL_REQUEST_TEMPLATE.md`](https://github.com/tempoCollaboration/TimeEvolvingMPO/blob/master/PULL_REQUEST_TEMPLATE.md)
to guarantee the quality of the package.
Therefore, before you submit the pull request, go through this checklist once
more. However, this list is there to *help* all contributors to produce high
quality code, not to deter you from contributing. If you can't tick some of the
boxes, feel free to submit the pull request anyway and report about it in the
pull request message.


### 10. Make changes to your pull request
You should find the pull request you created in section 9 on the
[pull request page](https://github.com/tempoCollaboration/TimeEvolvingMPO/pulls).
All the rest of the communication will happen there.

You can expect to hear back from a maintainer within one to two weeks. If your
contribution checks all the boxes in
`PULL_REQUEST_TEMPLATE.md`](https://github.com/tempoCollaboration/TimeEvolvingMPO/blob/master/PULL_REQUEST_TEMPLATE.md)
the maintainer will merge your changes into the project. If some aspects of the
contribution need to be changed, the maintainer might ask you if you'd be
willing to perform these changes. In this case you can simply make further
changes to your pull request branch and push them to your fork. This will
automatically update your pull request, and therfore you need not create a one.


**-- The End --**
