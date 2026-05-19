# TypeScript Frontend Integration Design

**Date:** 2026-05-19

## Summary

Integrate the existing `llm-ui` React/Vite/TypeScript app (currently in the `aipet` Turborepo) into this repo as a `ui/` subdirectory. Serve it in production via S3 + CloudFront at `llm.jwnwilson.co.uk`, deployed by GitHub Actions on push to `main`.

---

## Directory Structure

The llm-ui source moves into `ui/` at the repo root, kept exactly as-is:

```
llm_api/
  ui/                      ← extracted from aipet/apps/llm-ui
    src/
    public/
    package.json
    vite.config.ts
    tsconfig.json
    .env.local.example
  src/                     ← Python backend (unchanged)
  tests/
  pyproject.toml
  docker-compose.yml
```

No root-level `package.json`. The two toolchains (`uv` for Python, `npm` for the UI) stay fully independent — `pyproject.toml` and `package.json` do not interfere.

---

## Development Workflow

Each service runs independently:

```bash
# Terminal 1 — Python API
uv run uvicorn src.interactors.api.app:app --reload

# Terminal 2 — Vite dev server
cd ui && npm run dev
```

`vite.config.ts` is updated with a dev proxy to avoid CORS issues:

```ts
server: { proxy: { '/api': 'http://localhost:8000' } }
```

No changes to the Python side for local dev.

---

## Production — S3 + CloudFront

Vite builds static files; GitHub Actions syncs them to S3; CloudFront serves them at `llm.jwnwilson.co.uk`. No UI container or nginx needed.

### Terraform changes (`infra/terraform/`)

**New modules** — copy verbatim from `aipet/infra/terraform/modules/`:
- `modules/s3_static/` — private S3 bucket + CloudFront OAC distribution, SPA fallback (403/404 → `index.html`), HTTPS redirect, `PriceClass_100`
- `modules/acm/` — ACM certificate (must be in `us-east-1` for CloudFront)

**`main.tf`** — add:
```hcl
module "acm_ui" {
  source   = "./modules/acm"
  domain   = "llm.jwnwilson.co.uk"
  providers = { aws = aws.us_east_1 }  # CloudFront requires ACM certs in us-east-1
}

module "s3_ui" {
  source              = "./modules/s3_static"
  name                = "llm-api-ui"
  domain              = "llm.jwnwilson.co.uk"
  acm_certificate_arn = module.acm_ui.certificate_arn
}
```

Pass `module.s3_ui.bucket_arn`, `module.s3_ui.distribution_arn` into the `iam` module call.

**`modules/iam/main.tf`** — add a `ui_deploy` policy granting the GitHub Actions role:
- `s3:ListBucket`, `s3:PutObject`, `s3:DeleteObject` on the UI bucket
- `cloudfront:CreateInvalidation` on the UI distribution

Same pattern as `aipet/infra/terraform/modules/iam/main.tf`.

**`modules/dns/main.tf`** — add a CNAME record:
```
llm.jwnwilson.co.uk → <cloudfront_domain>
```
Accepts new input variable `ui_cf_domain` passed from `module.s3_ui.cloudfront_domain`.

**`outputs.tf`** — add:
- `ui_bucket_name` — set as `UI_BUCKET` GitHub secret after apply
- `ui_distribution_id` — set as `UI_CF_DISTRIBUTION_ID` GitHub secret after apply
- `ui_fqdn`

### GitHub Actions (`.github/workflows/deploy-ui.yml`)

Triggers on push to `main` when `ui/**` changes (`paths: ['ui/**']`).

Steps:
1. Checkout
2. `npm ci` (working dir: `ui/`)
3. `npm run build` (working dir: `ui/`)
4. `aws s3 sync ui/dist/ s3://$UI_BUCKET --delete`
5. `aws cloudfront create-invalidation --distribution-id $UI_CF_DISTRIBUTION_ID --paths "/*"`

Uses the existing OIDC role via `AWS_ROLE_ARN` secret — no new AWS credentials. Two new GitHub secrets required after `terraform apply`: `UI_BUCKET` and `UI_CF_DISTRIBUTION_ID`.

---

## Environment Config

`ui/.env.local.example` updated to reflect the new context:

```
VITE_AUTH0_DOMAIN=your-tenant.auth0.com
VITE_AUTH0_CLIENT_ID=your-client-id
VITE_AUTH0_AUDIENCE=https://api.llm.jwnwilson.co.uk
VITE_API_URL=http://localhost:8000
```

Production API URL is baked into the build via a `VITE_API_URL` environment variable set in the GitHub Actions workflow.

---

## Files Created / Modified

| Path | Action |
|------|--------|
| `ui/` | Created — copied from `aipet/apps/llm-ui` |
| `ui/vite.config.ts` | Modified — add dev proxy |
| `ui/.env.local.example` | Modified — update audience/URL |
| `infra/terraform/modules/s3_static/` | Created — copied from `aipet` |
| `infra/terraform/modules/acm/` | Created — copied from `aipet` |
| `infra/terraform/main.tf` | Modified — add `acm_ui`, `s3_ui` modules |
| `infra/terraform/modules/iam/main.tf` | Modified — add UI deploy policy |
| `infra/terraform/modules/iam/variables.tf` | Modified — add `ui_bucket_arn`, `ui_distribution_arn` vars |
| `infra/terraform/modules/dns/main.tf` | Modified — add CNAME record |
| `infra/terraform/modules/dns/variables.tf` | Modified — add `ui_cf_domain` var |
| `infra/terraform/outputs.tf` | Modified — add UI outputs |
| `.github/workflows/deploy-ui.yml` | Created — build + sync + invalidate |
