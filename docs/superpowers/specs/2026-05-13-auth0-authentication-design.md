# Auth0 Authentication Design

**Date:** 2026-05-13
**Status:** Approved

---

## Goal

Protect all LLM API endpoints with JWT-based authentication using Auth0 as the identity provider. The API acts as an OAuth2 resource server — it validates tokens issued by Auth0 but never issues them itself. A built-in login flow allows the service to be tested in isolation without a separate client application.

## Architecture

The API is a pure **resource server**. Clients (React app, other services) authenticate directly with Auth0 and pass the resulting access token as `Authorization: Bearer <token>`. The API validates the JWT on every protected request.

The implementation follows the existing adapter pattern in the codebase:

```
AuthPort (domain/ports.py)
  └── Auth0Adapter (adapters/auth/auth0.py)
        ↑ configured in lifespan (app.py) via deps.py
        ↓ called by require_auth (interactors/api/auth.py)
              ↓ applied as Depends() at router level
```

`GET /health` remains unauthenticated.

### Login flow (test isolation)

`GET /auth/login` redirects to Auth0 Universal Login using the OAuth2 authorisation code grant. After the user authenticates, Auth0 redirects to `GET /auth/callback`, which exchanges the code for tokens and returns the access token in plain text. This lets a developer log in and get a token to use in curl or Swagger without needing any other client application running.

---

## Components

### Domain layer (`src/domain/`)

**`models.py`** — add `UserContext`:
```python
class UserContext(BaseModel):
    user_id: str
    email: str | None = None
```

**`ports.py`** — add `AuthPort`:
```python
class AuthPort(ABC):
    @abstractmethod
    def authenticate(self, token: str) -> UserContext | None:
        """Validate the JWT and return a UserContext, or None if invalid."""
```

### Adapter layer (`src/adapters/auth/auth0.py`)

`Auth0Adapter(AuthPort)`:
- On first call, fetches JWKS from `https://{AUTH0_DOMAIN}/.well-known/jwks.json` and caches the keys in memory
- Validates the JWT: RS256 signature, expiry, issuer (`https://{AUTH0_DOMAIN}/`), audience (`AUTH0_AUDIENCE`)
- Returns `UserContext(user_id=payload["sub"], email=payload.get("email"))`
- Returns `None` on any validation failure (expired, tampered, wrong audience)

Dependencies: `PyJWT>=2.0`, `cryptography` (already likely present; needed for RS256 key loading).

### Interactor layer (`src/interactors/api/`)

**`auth.py`** — `require_auth` dependency:
- Reads `Authorization: Bearer <token>` header
- Calls `auth_port.authenticate(token)`
- Raises `HTTPException(401)` if the header is missing or the token is invalid

**`routes/login.py`** — login flow:
- `GET /auth/login` — redirects to Auth0 authorise endpoint with `response_type=code`, `scope=openid email`, audience, and callback URL
- `GET /auth/callback` — exchanges `code` for tokens via Auth0 token endpoint, returns `access_token` in plain text

**`deps.py`** — add singleton pattern (identical to existing store pattern):
```python
_auth_port: AuthPort | None = None

def get_auth() -> AuthPort: ...
def configure_auth(port: AuthPort) -> None: ...
```

**`app.py`** changes:
- Wire `Auth0Adapter` in lifespan and call `configure_auth`
- Replace `allow_origins=["*"]` with `CORS_ORIGINS` env var (`*` only when `APP_ENV=development`)
- Include the login router

**Route changes:**
- `routes/inference.py` — add `dependencies=[Depends(require_auth)]` to `/infer`
- `routes/models.py` — add `dependencies=[Depends(require_auth)]` at router level
- `routes/runs.py` — add `dependencies=[Depends(require_auth)]` at router level

---

## Environment Variables

| Variable | Required | Example | Purpose |
|---|---|---|---|
| `AUTH0_DOMAIN` | Yes | `your-tenant.auth0.com` | Auth0 tenant domain |
| `AUTH0_AUDIENCE` | Yes | `https://api.aipet.example.com` | API identifier registered in Auth0 |
| `AUTH0_CLIENT_ID` | Yes | `abc123` | Application client ID (for login flow) |
| `AUTH0_CLIENT_SECRET` | Yes | `secret` | Application client secret (for login flow) |
| `AUTH0_CALLBACK_URL` | Yes | `http://localhost:8000/auth/callback` | Redirect URI registered in Auth0 |
| `CORS_ORIGINS` | Yes (prod) | `https://app.example.com` | Comma-separated allowed origins |
| `APP_ENV` | No | `development` | Set to `development` for wildcard CORS |

---

## Error Handling

| Scenario | Response |
|---|---|
| Missing `Authorization` header | `401 {"detail": "Not authenticated"}` |
| Invalid / tampered token | `401 {"detail": "Invalid token"}` |
| Expired token | `401 {"detail": "Token expired"}` |
| Audience mismatch | `401 {"detail": "Invalid token"}` |
| JWKS endpoint unreachable | Warning logged; adapter retries on next request |

No 403 — auth is binary (valid token = full access).

---

## Testing

No live Auth0 tenant is required for any test.

**Unit tests:**
- `tests/unit/test_auth0_adapter.py` — mock JWKS endpoint; assert adapter accepts valid RS256 JWTs and rejects expired, tampered, and wrong-audience tokens
- `tests/unit/test_auth_dependency.py` — minimal FastAPI app with `FakeAuthAdapter`; assert 401 with no header, 401 with bad token, 200 with valid token

**Integration tests:**
- `tests/integration/test_auth.py` — `autouse` fixture injects `FakeAuthAdapter` that accepts any non-empty token; covers all protected routes returning 401 without a token and 200 with one; confirms `GET /health` returns 200 with no token
- `tests/integration/test_api.py` — existing tests get an `autouse` fixture that configures `FakeAuthAdapter` so they continue passing without modification to request code

---

## Auth0 Setup (one-time)

1. Create a new **API** in Auth0 dashboard — set the identifier (this becomes `AUTH0_AUDIENCE`)
2. Create a new **Regular Web Application** — copy client ID and secret
3. Add `{AUTH0_CALLBACK_URL}` to the application's **Allowed Callback URLs**
4. Enable **RS256** signing algorithm on the API (default)
