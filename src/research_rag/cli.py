"""One installable command for serving, ingesting, and evaluating the project."""

import argparse
import sys

import uvicorn

from research_rag import __version__
from research_rag.settings import get_settings


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        _root_parser().print_help()
        return
    if arguments[0] in {"-V", "--version"}:
        print(__version__)
        return

    command, command_args = arguments[0], arguments[1:]
    if command == "serve":
        _serve(command_args)
    elif command == "ingest":
        from research_rag.ingestion.service import main as ingest_main

        ingest_main(command_args)
    elif command == "eval":
        from research_rag.evaluation.runner import main as evaluation_main

        evaluation_main(command_args)
    else:
        _root_parser().error(f"unknown command: {command}")


def _serve(argv: list[str]) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="research-rag serve", description="Run the HTTP API and chat page."
    )
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    parser.add_argument("--reload", action="store_true", help="development only")
    args = parser.parse_args(argv)
    uvicorn.run(
        "research_rag.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )


def _root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-rag",
        description="Readable research-paper RAG",
        epilog=(
            "commands:\n"
            "  serve   run the API and chat page\n"
            "  ingest  download papers and build the index\n"
            "  eval    run the golden evaluation set\n\n"
            "Use 'research-rag COMMAND --help' for command options."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="store_true", help="show version")
    return parser
