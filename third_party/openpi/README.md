# OpenPI

Policy server and client based on [OpenPI](https://github.com/Physical-Intelligence/openpi),
adapted for activation collection and SAE interventions. The modified source is
bundled in this directory.

Event-SAE changes include `scripts/serve_policy.py`, `src/openpi/models/gemma.py`,
`src/openpi/models/pi0.py`,
`src/openpi/policies/policy.py`, `src/openpi/shared/download.py` (download error propagation),
and the modules in `src/openpi/sae_collection/`.

See the [main README](../../README.md) for usage.
Licenses: [Apache-2.0](LICENSE) and [Gemma](LICENSE_GEMMA.txt).
