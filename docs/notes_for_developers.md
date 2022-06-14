# Notes for Developers

## Versioning

Semantic versioning (i.e. a scheme that follows a `vMAJOR.MINOR.PATCH` format; see <https://semver.org> for details) is used for this project.  The single point of truth for the current production version is the last git tag on the main branch with a `v[0-9]*` format.

Changes to `PATCH` are handled by a GitHub workflow which increments this value and creates a new tag whenever a push occurs to the `main` branch.  This ensures that every commit on the `main` branch is assigned a unique version.  If a change to `MINOR` (for backward-compatible functionality changes) or to `MAJOR` (for breaking changes) is required then a new version tag should be manually added to the `main` branch through the GitHub UI.

## Releases

Releases are generated through the GitHub UI.  A GitHub workflow has been configured to do the following when a new release is produced:

* Run the tests for the project
* Ensure that the project builds
* Publish a new version of the code on [PyPI]().
* Rebuild the documentation on *Read the Docs*

To generate a new release, do the following:

* Navigate to the project's GitHub page, 
* Click on `Releases` in the sidebar,
* Click on `Create a new release` if this is the first release you have generatede, or `draft release` if this is a subsequent release,
* Click on `Choose a tag` and select the most recent version listed,
* Write some text describing the release, and
* Click `Publish Release`.

If the development environment is properly configured and all goes well (tests are passed, etc), then the following will happen:

* a new *GitHub* release will be generated;
* the release will be published on *PyPI*; and
* the documentation will be rebuilt on *Read the Docs*.

## Development Environment Set-up

This section details how to grab a copy of this code and configure it for development purposes.  In what follows, we will assume that you have already created *GitHub* and *Read the Docs* accounts for this purpose.  If not, first visit  <https://github.com> and/or <https://readthedocs.org> respectively to do so.

### The Code

A local copy of the code can be configured as follows:

1. Create a fork: 
	* navigate to the GitHub page hosting the project and click on the `fork` button at the top of the page;
	* Edit the details you want to have for the new repoitory; and
	* Press `Create fork`,
	* Generate a local 

2. If you want to work from a local clone:
	* First grab a local copy of the code (e.g. `git clone <url>` where `<url>` can be obtained by clicking on the green `Code` button on the project GitHub page);
	* Create a new GitHub repository for the account to host the code by logging into *GitHub* and clicking the `+` selector at the very top and selecting `New repository`;
	* Edit the form detailing the new repository and click `Create repository` at the bottom;
	* Add the new *GitHub* repo as a remote repository to your local copy via `git remote add origin <newurl>`, where `<newurl>` can be obtained by clicking on the green `Code` button on the new repository's page; and
	* Push the code to the new GitHub repository with `git push origin main`.

### *GitHub*

Configure your *GitHub* repository following the directions [here](https://docs.readthedocs.io/en/stable/integrations.html#github).

### *Read the Docs*

Navigate to your RTD Project page and "Import a Project".  Your GitHub project with its new
Webhook should appear in this list.  Import it.

The documentation for this project should now build automatically for you when you generate a
new release.
