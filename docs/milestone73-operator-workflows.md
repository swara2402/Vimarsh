# Milestone 73: Operator review and evidence workflows

## Human-review queue

The API provides a token-protected local SQLite queue for cases that need an operator's attention.

- `POST /v1/reviews` creates a pending case.
- `GET /v1/reviews?status=pending&limit=50` lists cases by priority and creation time.
- `POST /v1/reviews/{review_id}/claim` atomically assigns a pending case to an operator-supplied reviewer ID.
- `POST /v1/reviews/{review_id}/complete` records the adjudication and optional retrospective ground truth.
- `POST /v1/reviews/{review_id}/cancel` cancels a pending or claimed case with an audit reason.
- `GET /v1/reviews/{review_id}/events` returns lifecycle events after verifying the global SHA-256 hash chain.

Set `VIMARSH_REVIEWER_API_TOKEN` to enable the queue. Set `VIMARSH_REVIEW_QUEUE_DB` to select the SQLite database path. The default path is under `research/data/`.

Generate separate high-entropy local demo tokens rather than reusing passwords:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Place one generated value in `VIMARSH_REVIEWER_API_TOKEN` and a different generated value in `VIMARSH_EVIDENCE_API_TOKEN`. Keep the environment file private and do not commit real tokens.

Reviewer IDs are labels, not authenticated accounts. The shared token is not individual identity, MFA, or role-based access control. The queue is suitable only for a local controlled demo until a proper identity provider and authorization policy are configured.

## Evidence register

The evidence API stores operator-submitted evidence in a separate append-only SQLite database.

- `POST /v1/evidence` stores a manually submitted source name, reference, excerpt, metadata, and SHA-256 digest.
- `GET /v1/evidence/{case_id}` retrieves records and recomputes each digest to report `integrity_valid`.
- `POST /v1/evidence/resolve-doi` retrieves bibliographic metadata from the fixed Crossref works API and records it against a case.

Set `VIMARSH_EVIDENCE_API_TOKEN` to enable evidence endpoints. Set `VIMARSH_EVIDENCE_DB` to select the database path.

The Crossref adapter validates DOI syntax and only contacts the fixed Crossref API host. It records bibliographic metadata such as title, authors, publisher, year, and publication venue. It does not retrieve full text, verify a paper's claims, or determine whether the source is correct. Manual evidence remains operator-submitted and is not independently verified.

## Dashboard

The React dashboard includes separate in-memory token fields for the reviewer and evidence APIs, pending/claimed/completed queue filters, claim, completion and cancellation actions, audit-history inspection, manual evidence submission, DOI metadata lookup, and evidence retrieval with digest status.

Tokens are not written to local storage. Keep the API and dashboard bound to loopback for local demonstrations. Do not expose these shared-token endpoints publicly.

## Security and operational limitations

- API responses include conservative browser security headers; versioned API responses are marked `Cache-Control: no-store`.
- Review events and evidence rows use SQLite append-only triggers.
- The review audit log is hash-chained and verified before history is returned.
- Evidence hashes detect content changes relative to the stored digest, but do not prove source authenticity.
- SQLite is single-host storage, not a distributed database.
- Individual identities, per-user authorization, rate limiting, encrypted backups, retention controls, production secret management, external source verification, and deployment-specific security review remain required for production use.

## Validation

Run the backend suite from the repository root:

```bash
PYTHONPATH=backend pytest backend/tests -q
```

The operator workflow tests cover token gating, queue lifecycle transitions, audit-chain retrieval, append-only evidence storage, digest verification, Crossref adapter behavior, CORS headers, and API response headers.
