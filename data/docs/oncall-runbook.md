# On-Call Runbook

The on-call engineer acknowledges a page within 5 minutes and starts an incident channel for any SEV1 or SEV2.

## Severity levels

SEV1: the product is down or user funds or data are at risk. All hands, status page update within 15 minutes.
SEV2: a major feature is broken for many users. On-call plus the owning team.
SEV3: a minor feature is degraded or there is a workaround. Fix during working hours.

## First steps

Check the service dashboard, recent deploys and feature flag changes. If a deploy in the last hour lines up with the incident, roll it back first.

## After the incident

SEV1 and SEV2 incidents need a blameless postmortem within five working days, with action items that have owners and due dates.
