.. _release_procedure:

*****************
Release Procedure
*****************

This procedure is based off of `Astropy's
<https://docs.astropy.org/en/stable/development/releasing.html>`_, and
additionally uses information from the `PyPI packaging tutorial
<https://packaging.python.org/tutorials/packaging-projects/>`_.

Prerequisites
=============

To make releases, you will need

- The `twine <https://pypi.org/project/twine/>`_ package.
- An account on `PyPI <https://pypi.org/>`_.
- Collaborator status on Baseband's repository at ``mhvk/baseband`` to push new
  branches.
- An account on `Read the Docs <https://readthedocs.org/>`_ that has access
  to Baseband.
- Optionally, a `GPG signing key <https://help.github.com/articles/signing-commits-with-gpg/>`_
  associated with your GitHub account.  While releases do not need to be
  signed, we recommend doing so to ensure they are trustworthy.  To make a GPG
  key and associate it with your GitHub account, see the `Astropy documentation
  <https://docs.astropy.org/en/stable/development/releasing.html#creating-a-gpg-signing-key-and-a-signed-tag>`_.

Versioning
==========

Baseband follows the `semantic versioning specification <https://semver.org/>`_::

    major.minor.patch

where

- ``major`` number represents backward incompatible API changes.
- ``minor`` number represents feature updates to last major version.
- ``patch`` number represents bugfixes from last minor version.

Major and minor versions have their own release branches on GitHub that end
with "x" (eg. ``v1.0.x``, ``v1.1.x``), while specific releases are tagged
commits within their corresponding branch (eg. ``v1.1.0`` and ``v1.1.1`` are
tagged commits within ``v1.1.x``).

Procedure
=========

The first two steps of the release procedure are different for major and minor
releases than it is for patch releases.  Steps specifically for major/minor
releases are labelled "m", and patch ones labelled "p".

1m. Preparing major/minor code for release
------------------------------------------

We begin in the main development branch (the local equivalent to
``mhvk/baseband:master``).  First, check the following:

- **Ensure tests pass**.  Run the test suite by running ``tox -e test``
  in the Baseband root directory.
- **Update** ``CHANGES.rst``.  All merge commits to master since the last
  release should be documented (except trivial ones such as typo corrections).
  Since ``CHANGES.rst`` is updated for each merge commit, in practice it is
  only necessary to change the date of the release you are working on from
  "unreleased" to the current date.
- **Add authors and contributors to** ``AUTHORS.rst`` and ``.zenodo.json``.
  To list contributors, one can use::

      git shortlog -n -s -e

  This will also list contributors to astropy-helpers and the astropy
  template, who should not be added.  If in doubt, cross-reference with the
  authors of pull requests.

Once finished, ``git add`` any changes and make a commit::

    git commit -m "Finalizing changelog and author list for v<version>"

For major/minor releases, the patch number is ``0``.

Submit the commit as a pull request to master.

Make and enter a new release branch::

    git checkout -b v<version branch>.x

1p. Cherry-pick code for a patch release
----------------------------------------

We begin by checking out the appropriate release branch::

    git checkout v<version branch>.x

Bugfix merge commits are backported to this branch from master by way of ``git
cherry-pick``.  First, find the SHA hashes of the relevant merge commits in the
main development branch.  Then, for each::

    git cherry-pick -m 1 <SHA-1>

For more information, see `Astropy's documentation
<https://docs.astropy.org/en/stable/development/releasing.html#backporting-fixes-from-master>`_.

Once you have cherry-picked, check the following:

- **Ensure tests pass and documentation builds**.  Run the test suite by
  running ``tox -e test``, and build documentation by running
  ``tox -e build_docs``, in the Baseband root directory.
- **Update** ``CHANGES.rst``.  Typically, merge commits record their changes,
  including any backported bugfixes, in ``CHANGES.rst``.  Cherry-picking should
  add these records to this branch's ``CHANGES.rst``, but if not, manually
  add them before making the commit (and manually remove any changes not
  relevant to this branch). Also, change the date of the release you are
  working on from "unreleased" to the current date.

Commit your changes::

    git commit -m "Finalizing changelog for v<version>"

2. Tag the release
------------------

Tag the commit made in step 1 as::

    git tag -s v<version> -m "Tagging v<version>"

3. Clean and package the release
--------------------------------

Checkout the tag::

    git checkout v<version>

Clean the repository::

    git clean -dfx

and ensure the repository has the proper permissions::

    umask 0022
    chmod -R a+Xr .

Finally, package the release's source code::

    python3 -m build --sdist --outdir dist .

4. Test the release
-------------------

We now test installing and running Baseband in clean virtual environments, to
ensure there are no subtle bugs that come from your customized development
environment. Before creating the virtualenvs, we recommend checking if the
``$PYTHONPATH`` environmental variable is set.  If it is, set it to a null
value (in bash, ``PYTHONPATH=``) before proceeding.

To create the environments and test in it::

    python3 -m venv test_release
    source test_release/bin/activate
    pip install dist/baseband-*.tar.gz
    pip install pytest-astropy
    cd ~/
    pytest --pyargs baseband
    deactivate
    cd -

If the test suite raises any errors (at this point, likely dependency issues),
delete the release tag::

    git tag -d v<version>

For a major/minor release, delete the ``v<version branch>.x`` branch as well.
Then, make the necessary changes directly on the main development branch.  Once
the issues are fixed, repeat steps 2 - 6.

If the tests succeed, you may optionally re-run the cleaning and packaging code
above following the tests::

    git clean -dfx
    umask 0022
    chmod -R a+Xr .
    python3 -m build --sdist --outdir dist .

You may optionally sign the source as well::

    gpg --detach-sign -a dist/baseband-*.tar.gz

5. Publish the release on GitHub
--------------------------------

If you are working a major/minor release, first push the branch to upstream
(assuming upstream is ``mhvk/baseband``)::

    git push upstream v<version branch>.x

Push the tag to GitHub as well::

    git push upstream v<version>

Go to the ``mhvk/baseband`` `Releases section
<https://github.com/mhvk/baseband/releases>`_ and check the Tags part.  To
publish a release, click on the ``...`` on the right for the tag you just pushed.
This takes you to a form where you can customize the release title and description.
Leave the title blank, in which case it is set to "v<version>"; for
the description, paste in the relevant parts of ``CHANGES.rst``.
Finally, click on "Publish release".  This takes you back to Releases.

The Baseband GitHub repo `automatically updates
<https://guides.github.com/activities/citable-code/>`_ Baseband's `Zenodo
<https://doi.org/10.5281/zenodo.1214268>`_ repository for each published release.
Check if your release has made it to Zenodo by clicking the badge in
``README.rst``.

6. (Optional) test uploading the release
----------------------------------------

PyPI provides a test environment to safely try uploading new releases.  To take
advantage of this, use::

    twine upload --repository-url https://test.pypi.org/legacy/ dist/baseband-<version>*

To test if this was successful, create a new virtualenv as above::

    virtualenv --no-site-packages --python=python3 pypitest

Then (``pip install pytest-astropy`` comes first because ``test.pypi`` does not
contain recent versions of Astropy)::

    source <name_of_virtualenv>/bin/activate
    pip install pytest-astropy
    pip install --index-url https://test.pypi.org/simple/ baseband
    pytest --pyargs baseband
    deactivate

7. Upload to PyPI
-----------------

Finally, upload the package to PyPI::

    twine upload dist/baseband-<version>*

8. Check if Readthedocs has updated
-----------------------------------

Go to `Read the Docs <https://readthedocs.org/>`_ and check that the
``stable`` version points to the latest stable release.  Each minor release has
its own version as well, which should be pointing to its latest patch release.

9m. Clean up master
--------------------

In the main development branch, add the next major/minor release to
``CHANGES.rst``.  Make a commmit::

    git commit -m "Add v<next major/minor version> to the changelog."

Then submit a pull request to master.

9p. Update CHANGES.rst on master
---------------------------------

Change the release date of the patch release in ``CHANGES.rst`` on master to
the current date, then::

    git commit -m "Added release date for v<version> to the changelog."

(Alternatively, ``git cherry-pick`` the changelog fix from the release branch
back to the main development one.)

Then submit a pull request to master.
