# CI Pipeline

Every pull request runs the `pr-checks` pipeline: lint, unit tests, integration tests and a build. The `main` pipeline adds a container image build and a staging deploy.

## Stages and time limits

Lint (ruff and eslint) must finish in 3 minutes. Unit tests have a 10 minute limit and integration tests a 20 minute limit. A stage that exceeds its limit is cancelled and reported as a timeout.

## Required checks

A pull request cannot merge until lint, unit tests and integration tests are green. The build stage is advisory on pull requests and required on `main`.

## Caching

Python dependencies are cached by the hash of `requirements.lock`. If a dependency install fails, clear the cache from the pipeline page and rerun before changing the lock file.
