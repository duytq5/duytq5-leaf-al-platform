"""Entry point inside the worker image: `python -m worker.run_job`.

The agent mounts a folder at /out holding job.json (the job without its task
token, which stays with the agent). This runs the handler for the job's type
and writes its result to /out/result.json; a non-zero exit fails the job.
"""

import json
import sys
from pathlib import Path

from common.jobs import parse_job
from worker.jobs import HANDLERS

OUT = Path("/out")


def main(out: Path = OUT) -> int:
    data = json.loads((out / "job.json").read_text())
    data["task_token"] = "held-by-the-agent"  # parse_job needs one; never used here
    job = parse_job(json.dumps(data))
    result = HANDLERS[job.type](job)
    (out / "result.json").write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
