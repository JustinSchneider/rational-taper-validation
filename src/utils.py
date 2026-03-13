"""Shared utilities: logging setup, project paths, and common helpers."""

import logging
from pathlib import Path


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: "str | Path | bool | None" = None,
    clear_logs: bool = False,
) -> logging.Logger:
    """Create a consistently-formatted logger.

    Args:
        name: Logger name (typically __name__ of calling module).
        level: Logging level (default INFO).
        log_file: If a path (str/Path), write logs to that file.
                  If True, auto-generate ``logs/{name}.log`` under project root.
                  If None/False, console only (backward-compatible default).
        clear_logs: If True, truncate the log file on each setup (mode='w').
                    If False, append (mode='a').  Ignored when *log_file* is falsy.

    Returns:
        Configured logger with console handler and optional file handler.
    """
    logger = logging.getLogger(name)

    # Clear existing handlers (safe for Jupyter re-runs)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (always present)
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (optional)
    if log_file:
        if log_file is True:
            log_dir = get_project_root() / "logs"
            log_dir.mkdir(exist_ok=True)
            log_path = log_dir / f"{name.replace('.', '_')}.log"
        else:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

        file_mode = "w" if clear_logs else "a"
        fh = logging.FileHandler(log_path, mode=file_mode, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Walks up from this file until it finds CLAUDE.md.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "CLAUDE.md").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root (no CLAUDE.md found)")


def get_db_path() -> Path:
    """Return the path to the SQLite database file."""
    return get_project_root() / "data" / "processed" / "galaxy_dynamics.db"
