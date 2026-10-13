"""One code asset for the Python Lambdas: the repo's packages plus the
third-party libraries they import, for the Lambda runtime (Amazon Linux 2023, x86_64).

`cdk deploy` builds it on your machine with uv (no Docker needed): uv
downloads the Linux wheels whatever OS you run on. Without uv, CDK falls back
to building it in the Lambda Python image with Docker. Library versions are
the ones in uv.lock, so the Lambdas run what the tests ran.
"""

import hashlib
import shutil
import subprocess
import tomllib
from pathlib import Path

import jsii
from aws_cdk import AssetHashType, BundlingOptions, ILocalBundling
from aws_cdk import aws_lambda as lambda_

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = lambda_.Runtime.PYTHON_3_12
PACKAGES = ("common", "db", "lambdas", "selection")
# What the round Lambdas import; boto3 comes with the Lambda runtime.
LIBRARIES = ("numpy", "psycopg", "psycopg-binary", "pyarrow", "pydantic", "pyyaml")


def pinned_requirements(lock: Path = ROOT / "uv.lock") -> list[str]:
    versions = {p["name"]: p["version"] for p in tomllib.loads(lock.read_text())["package"]}
    return [f"{name}=={versions[name]}" for name in LIBRARIES]


# Lambda unzips code into at most 250 MB, and pyarrow alone is ~150 MB. These
# parts are never loaded by the Lambdas (Arrow Flight, C headers, test suites).
PRUNE = (
    "pyarrow/include",
    "pyarrow/tests",
    "pyarrow/libarrow_flight.so.*",
    "pyarrow/_flight.*.so",
    "pyarrow/libarrow_python_flight.so*",
    "pyarrow/*.pyx",
    "pyarrow/*.pxd",
    "numpy/*/tests",
    "numpy/_core/include",
    "bin",
)


def _prune(out: Path) -> None:
    for pattern in PRUNE:
        for path in out.glob(pattern):
            shutil.rmtree(path) if path.is_dir() else path.unlink()


def _copy_packages(out: Path) -> None:
    for pkg in PACKAGES:
        shutil.copytree(
            ROOT / pkg, out / pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )


def source_hash() -> str:
    """Changes when the Lambdas' code or libraries change, and only then, so
    `cdk deploy` updates the functions exactly when it should."""
    h = hashlib.sha256("\n".join(pinned_requirements()).encode())
    for pkg in PACKAGES:
        for path in sorted((ROOT / pkg).rglob("*.py")):
            h.update(str(path.relative_to(ROOT)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


@jsii.implements(ILocalBundling)
class _UvBundling:
    def try_bundle(self, output_dir: str, _options) -> bool:
        uv = shutil.which("uv")
        if uv is None:
            return False
        out = Path(output_dir)
        subprocess.run(
            [
                uv, "pip", "install", "--quiet", "--target", str(out),
                "--python-platform", "x86_64-manylinux_2_28", "--python-version", "3.12",
                "--only-binary", ":all:", *pinned_requirements(),
            ],
            check=True,
        )  # fmt: skip
        _prune(out)
        _copy_packages(out)
        return True


def code() -> lambda_.Code:
    requirements = " ".join(f"'{r}'" for r in pinned_requirements())
    copy = " ".join(f"/asset-input/{p}" for p in PACKAGES)
    prune = " ".join(f"/asset-output/{p}" for p in PRUNE)
    return lambda_.Code.from_asset(
        str(ROOT),
        asset_hash_type=AssetHashType.CUSTOM,
        asset_hash=source_hash(),
        # Copied for the Docker fallback only; keep the big local folders out.
        exclude=[".git", ".venv", "cdk.out", "**/__pycache__", "deploy/ec2/data", "*.crt"],
        bundling=BundlingOptions(
            image=RUNTIME.bundling_image,
            local=_UvBundling(),
            command=[
                "bash",
                "-c",
                f"pip install --quiet --target /asset-output --only-binary :all: {requirements}"
                f" && rm -rf {prune} && cp -r {copy} /asset-output/",
            ],
        ),
    )
