"""One-time OAuth setup — opens a browser and saves a token for the Colab adapter."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/drive"]
DEFAULT_TOKEN = os.path.expanduser("~/.config/aipet/google_token.json")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Authorise Google Drive access for the Colab adapter.")
    parser.add_argument(
        "--client-secrets",
        default=os.environ.get("GOOGLE_OAUTH_CLIENT_SECRETS", "client_secrets.json"),
        dest="client_secrets",
        help="Path to the OAuth 2.0 Desktop app client_secrets.json downloaded from GCP Console",
    )
    parser.add_argument(
        "--token-file",
        default=os.environ.get("GOOGLE_OAUTH_TOKEN_FILE", DEFAULT_TOKEN),
        dest="token_file",
        help="Where to save the token (default: ~/.config/aipet/google_token.json)",
    )
    args = parser.parse_args(argv)

    if not Path(args.client_secrets).exists():
        raise SystemExit(
            f"client_secrets.json not found at {args.client_secrets!r}.\n"
            "Download it from GCP Console → APIs & Services → Credentials → "
            "OAuth 2.0 Client IDs → your Desktop app → Download JSON."
        )

    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(args.client_secrets, SCOPES)
    creds = flow.run_local_server(port=0)

    token_path = Path(args.token_file)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json())
    print(f"Token saved to {token_path}")
    print("You can now run: make colab-train-fast")


if __name__ == "__main__":
    main()
