# Code Review Guidelines

Every change needs one approving review from a code owner. Changes to authentication, payments or infrastructure need two approvals.

## Size

Keep pull requests under 400 changed lines where possible. Large refactors should be split into a mechanical change and a behaviour change.

## What reviewers check

Reviewers check correctness, tests for new behaviour, error handling, logging of failures, and that no secrets or personal data appear in code or logs.

## Response time

Reviewers should give a first response within one working day. Authors resolve or reply to every comment before merging.

## AI-generated code

AI coding assistants may be used, but the author is responsible for every line. AI-generated code gets the same review as any other code, and the pull request description must say which parts were generated.
