# Feature Flags

New user-facing behaviour ships behind a feature flag so it can be turned off without a deploy.

## Naming

Flags are named `team.feature_name`, for example `payments.instant_refunds`.

## Rollout

Roll out in steps: internal users, 1%, 10%, 50%, 100%. Watch error rates for at least one hour between steps.

## Cleanup

Remove a flag and its dead code within 30 days of reaching 100%. Stale flags are listed in the weekly engineering report.
