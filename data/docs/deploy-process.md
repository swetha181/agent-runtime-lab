# Deploy Process

Production deploys run from the `main` branch through the `deploy-prod` pipeline, Monday to Thursday between 09:00 and 16:00 local time. Friday deploys need approval from the on-call engineering manager.

## Canary

Every deploy starts with a 10% canary for 15 minutes. The canary is promoted automatically if the error rate stays below 1% and p95 latency does not rise by more than 20% compared with the previous release.

## Rollback

To roll back, run the `rollback` job on the previous successful release tag. Rollback takes about 4 minutes. Roll back first and investigate later whenever a canary breaches its error budget.

## Database migrations

Migrations must be backward compatible with the previous release so that a rollback never needs a down-migration. Destructive changes ship in a later release, after the code no longer uses the column.
