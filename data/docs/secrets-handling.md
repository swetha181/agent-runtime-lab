# Secrets Handling

Secrets such as API keys, database passwords and signing keys live in the secrets manager, never in code, config files or CI logs.

## Access

Services read secrets at start-up through their service identity. Engineers do not copy production secrets to laptops.

## If a secret leaks

If a secret is committed or printed in a log, rotate it immediately, then remove it from history. Rotation comes first because the secret must be treated as compromised the moment it leaves the secrets manager.

## CI

CI jobs receive only the secrets they need, scoped to the pipeline. Logs mask known secret values automatically.
