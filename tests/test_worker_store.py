"""The worker's S3 store sets the Content-Type the app relies on."""

from worker.compute import S3Store, content_type


class FakeS3:
    def __init__(self) -> None:
        self.puts: list[dict] = []

    def put_object(self, **kwargs) -> None:
        self.puts.append(kwargs)


def test_json_is_served_as_json_and_the_rest_as_bytes():
    assert content_type("edge/rocole-fastvit/2/manifest.json") == "application/json"
    assert content_type("edge/rocole-fastvit/2/model.onnx") == "application/octet-stream"
    assert content_type("scores/round/scores.parquet") == "application/octet-stream"


def test_put_sends_the_content_type():
    s3 = FakeS3()
    S3Store(s3).put("s3://models/edge/rocole-fastvit/2/manifest.json", b"{}")
    (put,) = s3.puts
    assert put == {
        "Bucket": "models",
        "Key": "edge/rocole-fastvit/2/manifest.json",
        "Body": b"{}",
        "ContentType": "application/json",
    }
