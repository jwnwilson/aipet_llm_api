"""CLI entry point to apply Alembic migrations."""

import sys

from adapters.database.engine import make_engine, run_migrations


def main() -> None:
    engine = make_engine()
    run_migrations(engine)
    print("Migrations applied.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        sys.exit(1)
