"""Logging helpers for the Local AgentOps application."""

import logging


def configure_logging(log_level: str = "INFO") -> None:
    """Configure the root logging system for local development.

    Args:
        log_level: Human-readable logging level such as ``INFO`` or ``DEBUG``.

    The function applies a single consistent log format and can be safely called
    multiple times during startup because ``basicConfig`` will no-op after the
    first successful configuration in most standard runtime flows.
    """
    # Normalize the requested level to an integer understood by the logging
    # module while falling back to INFO if the value is unexpected.
    resolved_level = getattr(logging, log_level.upper(), logging.INFO)

    # Initialize the root logging configuration with a concise local format.
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger instance.

    Args:
        name: Fully qualified logger name, usually ``__name__`` from a module.

    Returns:
        A standard library ``Logger`` configured under the shared root logger.
    """
    # Delegate logger creation to the standard logging registry.
    logger = logging.getLogger(name)

    # Return the resolved logger to the caller.
    return logger
