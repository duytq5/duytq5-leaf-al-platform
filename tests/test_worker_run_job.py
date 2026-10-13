"""`python -m worker.run_job` inside the worker image: reads /out/job.json,
runs the handler for its type, writes /out/result.json."""

import json
import uuid

from worker import run_job

COMMIT = "a" * 40


def test_runs_the_handler_and_writes_the_result(tmp_path, monkeypatch):
    job = {
        "job_id": str(uuid.uuid4()),
        "type": "export",
        "image": f"ghcr.io/duytq5/duytq5-leaf-al-platform-worker:{COMMIT}",
        "spec": {
            "model_uri": "models:/m/1",
            "output_prefix": "edge/m/1/",
            "labels": [{"code": "a", "display_name": "A"}, {"code": "b", "display_name": "B"}],
        },
    }
    (tmp_path / "job.json").write_text(json.dumps(job))
    seen = []
    monkeypatch.setitem(run_job.HANDLERS, "export", lambda j: seen.append(j) or {"done": True})
    assert run_job.main(tmp_path) == 0
    assert json.loads((tmp_path / "result.json").read_text()) == {"done": True}
    assert seen[0].spec.output_prefix == "edge/m/1/"
