# Data card

Project license: [MIT](../LICENSE). Dataset terms remain those of the pinned
upstream dataset.

The study uses Salesforce/xlam-function-calling-60k at revision
`26d14ebfe18b1f7b524bd39b404b50af5dc97866`. No prepared data is included in
the repository. Normalization accepts strict JSON schemas and one gold answer,
then records rendered length. Smoke, day1, core, and nested 1k/5k/10k pools
use records at or below 512 tokens. Only the 20k endpoint can extend the 10k
pool with records at or below 2,048 tokens.

The planned train sizes are 128, 2,000, 10,000, and 20,000 records. IID
validation and test locks are shared. Schema-OOD records are reserved before
IID allocation and stay outside locked API-prefix and schema-fingerprint
components. Duplicate components use stable SHA-256 ranking.

Raw records and the deterministic audit queue remain local and ignored.
Tracked dataset evidence is limited to sanitized content-addressed manifests.
No data audit, prepared dataset, or result is currently claimed as complete.
