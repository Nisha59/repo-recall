"""Run the API and UI locally: `python -m repo_recall.api [--host H] [--port P]`."""

import argparse

import uvicorn


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Serve the RepoRecall API and UI.")
    parser.add_argument("--host", default="127.0.0.1", help="interface to listen on (0.0.0.0 for all)")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    uvicorn.run("repo_recall.api.app:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
