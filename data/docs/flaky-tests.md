# Flaky Test Policy

A flaky test is a test that fails and then passes on retry without any code change.

## Rerun rule

You may rerun a failed CI job once. If the same test fails twice in a row it is treated as a real failure, not a flake.

## Quarantine

A test that flakes three or more times in seven days is quarantined with the `@pytest.mark.quarantine` marker and a ticket is opened for the owning team. Quarantined tests still run nightly but do not block merges. A quarantined test must be fixed or deleted within 14 days.

## Common causes

Most flakes come from network calls to real services, shared state between tests, unordered collections compared as lists, and timing assumptions such as fixed `sleep` calls. Mock external services and use explicit waits instead.
