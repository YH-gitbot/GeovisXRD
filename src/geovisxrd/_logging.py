# src/geovisxrd/_logging.py
"""
Package-level logging configuration for GeoVisXRD.

Usage (application code only — never call from library internals):
    import geovisxrd
    geovisxrd.setup_logging()          # INFO and above to stderr
    geovisxrd.setup_logging(level=logging.DEBUG)

Console output format:
    [GeoVisXRD] INFO: Geo table built — 500 rows, coords=(Latitude, Longitude), 8 SHAP columns
    [GeoVisXRD] WARNING: geopandas not installed — boundary layer skipped.
    [GeoVisXRD] ERROR: CSV save failed: ...
"""

import logging

_PACKAGE = "geovisxrd"

# Third-party loggers that produce noisy INFO output by default.
_THIRD_PARTY_LOGGERS = (
    "lightgbm",
    "xgboost",
    "shap",
    "numexpr",
)


class _GeoVisXRDFormatter(logging.Formatter):
    """Formats every record as  [GeoVisXRD] LEVEL: message."""

    _LABELS = {
        logging.DEBUG:    "DEBUG",
        logging.INFO:     "INFO",
        logging.WARNING:  "WARNING",
        logging.ERROR:    "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        level = self._LABELS.get(record.levelno, record.levelname)
        msg = record.getMessage()
        result = f"[GeoVisXRD] {level}: {msg}"

        # PREVIOUSLY BROKEN: the original code returned f"[GeoVisXRD] {level}: {record.getMessage()}"
        # directly. record.getMessage() only returns the message text; it never appends exception
        # info. As a result, any logger.exception() or logger.error(..., exc_info=True) call would
        # silently drop the traceback. The block below replicates what logging.Formatter.format()
        # does for exc_info and stack_info so that tracebacks appear correctly.
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            result += "\n" + record.exc_text
        if record.stack_info:
            result += "\n" + self.formatStack(record.stack_info)

        return result


def _suppress_third_party() -> None:
    """Raise the log floor of noisy third-party libraries to WARNING.

    Called once at package import.  Safe to call from library code because it
    only *raises* the effective level — it never installs handlers or affects
    the root logger.

    LIMITATION — Python logger vs native C++ output:
    This function only suppresses messages that flow through Python's logging
    module.  LightGBM and XGBoost also write training-progress lines (e.g.
    "[1]\ttraining's rmse: 0.456" or "[0] train-rmse:0.500") directly from
    their C++ cores to stdout/stderr, completely bypassing Python logging.
    Those lines are silenced by the verbosity=0 / verbose=-1 constructor
    defaults set in model_factory.py — NOT by this function.

    LIMITATION — tqdm and SHAP progress bars:
    tqdm writes its progress bars directly to sys.stderr and is not routed
    through Python logging.  Setting logging.getLogger("shap").setLevel(WARNING)
    has no effect on tqdm output produced inside SHAP or in explainer.py.
    The only way to suppress tqdm bars is to pass disable=True to each tqdm()
    call, or set tqdm.tqdm.disable = True globally before computation.
    """
    for name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the geovisxrd namespace.

    All internal modules must obtain their logger through this function instead
    of calling logging.getLogger(__name__) directly.  Using a single entry
    point means future cross-cutting concerns (custom filters, rate limiting,
    structured-log adapters) can be applied in one place without touching every
    module.

    Because every module's __name__ already starts with "geovisxrd.", the
    returned logger is automatically a child of the package-level logger
    configured by setup_logging() and inherits its handler and level.

    Usage (inside any geovisxrd submodule):
        from .._logging import get_logger   # one level up from any subpackage
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)


def setup_logging(level: int = logging.INFO, handler: logging.Handler = None) -> None:
    """Configure GeoVisXRD's logger with the [GeoVisXRD] PREFIX format.

    Call this **once** from application or notebook code.  Calling it multiple
    times is safe — subsequent calls are no-ops if a real handler is already
    attached.

    Parameters
    ----------
    level   : int  — minimum log level, e.g. logging.DEBUG (default INFO)
    handler : logging.Handler | None
        Custom handler to attach.  Defaults to a StreamHandler writing to
        stderr.  The [GeoVisXRD] formatter is applied regardless.
    """
    pkg_logger = logging.getLogger(_PACKAGE)

    # PREVIOUSLY BROKEN: the original guard was `if pkg_logger.handlers: return`.
    # __init__.py unconditionally adds a NullHandler to this logger on import
    # (standard library practice per the logging HOWTO).  That means
    # pkg_logger.handlers is NEVER empty by the time user code calls
    # setup_logging(), so the old guard always returned early — making
    # setup_logging() a permanent no-op and the formatted output never appeared.
    #
    # Fix: ignore NullHandler entries; only treat real handlers as "already configured".
    real_handlers = [h for h in pkg_logger.handlers if not isinstance(h, logging.NullHandler)]
    if real_handlers:
        return  # already configured — avoid duplicate handlers

    if handler is None:
        handler = logging.StreamHandler()

    handler.setFormatter(_GeoVisXRDFormatter())
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(level)
    # Prevent records from also reaching the root logger when setup_logging()
    # has been called — avoids duplicate lines if the app also configures root.
    pkg_logger.propagate = False
