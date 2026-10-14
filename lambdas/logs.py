"""One log line per handler call, with the ids that tie it to a round or release.

The Lambdas log JSON (infra/logs.py): the ids become fields, and they are in
the message too, so `al logs <id>` finds the line either way. Only the keys
below are logged; event bodies can hold task tokens.
"""

import functools
import logging
from collections.abc import Callable

log = logging.getLogger("lambdas")
log.setLevel(logging.INFO)

IDS = ("round_id", "job_id", "image_id", "dataset", "model", "version", "commit_sha")


def traced(handler: Callable) -> Callable:
    @functools.wraps(handler)
    def wrapper(event, context):
        ids = {k: event[k] for k in IDS if k in event} if isinstance(event, dict) else {}
        text = " ".join(f"{k}={v}" for k, v in ids.items())
        log.info("%s %s", handler.__name__, text, extra=ids)
        try:
            return handler(event, context)
        except Exception:
            log.exception("failed %s %s", handler.__name__, text, extra=ids)
            raise

    wrapper.traced = True
    return wrapper
