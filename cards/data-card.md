# Data card

Project license: [MIT](../LICENSE). Dataset terms remain those of the pinned
upstream dataset.

The study uses Salesforce/xlam-function-calling-60k at revision
`26d14ebfe18b1f7b524bd39b404b50af5dc97866`. Normalization accepts strict JSON
schemas, one gold answer, and a 512-token rendering limit. Duplicate components
are allocated by stable SHA-256 ranking.

The published train sizes are 128, 2,000, 10,000, and 20,000 records. Core
validation and test locks remain stable. Schema-OOD test records stay outside
all full-training and locked-validation API-prefix and schema-fingerprint
components.

Raw records and the deterministic audit queue remain local and ignored. Only
aggregate audit findings belong in tracked artifacts. Results remain preliminary
until final runs include commit and provenance metadata.
