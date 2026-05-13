"""Auth0 OAuth2 authorisation-code login — for development and isolated testing."""

from __future__ import annotations

import os
import urllib.parse

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, RedirectResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/login")
def login() -> RedirectResponse:
    params = {
        "response_type": "code",
        "client_id": os.environ["AUTH0_CLIENT_ID"],
        "redirect_uri": os.environ["AUTH0_CALLBACK_URL"],
        "audience": os.environ["AUTH0_AUDIENCE"],
        "scope": "openid email",
    }
    url = (
        f"https://{os.environ['AUTH0_DOMAIN']}/authorize?"
        + urllib.parse.urlencode(params)
    )
    return RedirectResponse(url=url)


@router.get("/callback")
# IMPORTANT: This endpoint must be served over HTTPS in production.
# The `code` query parameter arrives in the URL, which means it's visible in
# browser history, server access logs, and Referer headers. HTTPS prevents
# interception; HTTP exposes the code to network observers.
def callback(code: str) -> PlainTextResponse:
    domain = os.environ["AUTH0_DOMAIN"]
    resp = httpx.post(
        f"https://{domain}/oauth/token",
        json={
            "grant_type": "authorization_code",
            "client_id": os.environ["AUTH0_CLIENT_ID"],
            "client_secret": os.environ["AUTH0_CLIENT_SECRET"],
            "code": code,
            "redirect_uri": os.environ["AUTH0_CALLBACK_URL"],
        },
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Auth0 token exchange failed")
    access_token = resp.json().get("access_token", "")
    return PlainTextResponse(
        f"Access token (copy this for API calls):\n\n{access_token}\n\n"
        "Use with:  Authorization: Bearer <token>"
    )
