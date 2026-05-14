#!/usr/bin/env python3
"""Update AWS credentials in .env from the current AWS profile.

Reads credentials via `aws configure export-credentials` (supports both
static IAM keys and temporary STS/SSO sessions) and updates .env in-place,
uncommenting any commented-out credential lines as needed.
"""

import re
import subprocess
import sys
from pathlib import Path


def main() -> None:
    result = subprocess.run(
        ["aws", "configure", "export-credentials", "--format", "env-no-export"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    creds = dict(
        line.split("=", 1)
        for line in result.stdout.strip().splitlines()
        if "=" in line
    )

    env_path = Path(".env")
    text = env_path.read_text() if env_path.exists() else ""

    for key, value in creds.items():
        if re.search(rf"^#?{key}=", text, re.M):
            text = re.sub(rf"^#?{key}=.*", f"{key}={value}", text, flags=re.M)
        else:
            text += f"\n{key}={value}"

    env_path.write_text(text)
    print("Updated:", ", ".join(creds))


if __name__ == "__main__":
    main()
