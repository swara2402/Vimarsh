# Milestone 65: Capability-aware research dashboard

## Delivered implementation

- The React dashboard now reads `GET /v1/actions/capabilities` from the running API.
- It displays live availability and mode/reason for accept, alternate-model inference, human review, additional evidence, and abstention.
- The alternate-model availability control is disabled when the trusted runtime reports no alternate model.
- Decision requests include alternate-model availability and estimated cost only when the API capability says the path is configured.
- The panel explicitly displays the execution boundary returned by the API, keeping recommendation-only behavior separate from actual local inference.
- Capability discovery is independent from model-schema loading so an informational endpoint failure does not prevent the primary dashboard connection attempt.

## Safety and interpretation

- Capability discovery reports configuration, not action quality or real-world effectiveness.
- Human review and additional evidence remain unavailable until real providers are integrated.
- Alternate-model availability means a non-primary model exists in the trusted local bundle; it does not mean the system has selected a better prediction.
- The dashboard continues to require an explicit user action to run alternate models and does not automatically replace the primary result.
