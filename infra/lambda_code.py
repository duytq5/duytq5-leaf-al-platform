"""Code assets for the Python Lambdas: some of the repo's packages plus the
third-party libraries they import, for the Lambda runtime (Amazon Linux 2023, x86_64).

Two bundles: ROUND for the al-round Lambdas (Postgres, Parquet, the selection
strategies) and API for the FastAPI Lambda, which stays small.

`cdk deploy` builds them on your machine with uv (no Docker needed): uv
downloads the Linux wheels whatever OS you run on. Without uv, CDK falls back
to building them in the Lambda Python image with Docker. Library versions are
the ones in uv.lock, so the Lambdas run what the tests ran.
"""

import hashlib
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

import jsii
from aws_cdk import AssetHashType, BundlingOptions, ILocalBundling
from aws_cdk import aws_lambda as lambda_

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = lambda_.Runtime.PYTHON_3_12


def lock_versions(lock: Path = ROOT / "uv.lock") -> dict[str, str]:
    return {p["name"]: p["version"] for p in tomllib.loads(lock.read_text())["package"]}


@dataclass(frozen=True)
class Bundle:
    packages: tuple[str, ...]
    # Top-level libraries, pinned to uv.lock; boto3 comes with the Lambda runtime.
    libraries: tuple[str, ...]
    # Parts of the libraries the Lambdas never load (globs under the bundle).
    prune: tuple[str, ...] = ()

    def requirements(self) -> list[str]:
        versions = lock_versions()
        return [f"{name}=={versions[name]}" for name in self.libraries]

    def source_hash(self) -> str:
        """Changes when the bundle's code or libraries change, and only then,
        so `cdk deploy` updates the functions exactly when it should."""
        h = hashlib.sha256("\n".join(self.requirements() + list(self.prune)).encode())
        for pkg in self.packages:
            for path in sorted((ROOT / pkg).rglob("*.py")):
                h.update(str(path.relative_to(ROOT)).encode())
                h.update(path.read_bytes())
        return h.hexdigest()

    def build(self, out: Path, uv: str) -> None:
        subprocess.run(
            [
                uv, "pip", "install", "--quiet", "--target", str(out),
                "--python-platform", "x86_64-manylinux_2_28", "--python-version", "3.12",
                "--only-binary", ":all:", *self.requirements(),
            ],
            check=True,
        )  # fmt: skip
        for pattern in self.prune:
            for path in out.glob(pattern):
                shutil.rmtree(path) if path.is_dir() else path.unlink()
        for pkg in self.packages:
            shutil.copytree(
                ROOT / pkg, out / pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )

    def code(self) -> lambda_.Code:
        requirements = " ".join(f"'{r}'" for r in self.requirements())
        copy = " ".join(f"/asset-input/{p}" for p in self.packages)
        prune = " ".join(f"/asset-output/{p}" for p in self.prune)
        return lambda_.Code.from_asset(
            str(ROOT),
            asset_hash_type=AssetHashType.CUSTOM,
            asset_hash=self.source_hash(),
            # Copied for the Docker fallback only; keep the big local folders out.
            exclude=[".git", ".venv", "cdk.out", "**/__pycache__", "deploy/ec2/data", "*.crt"],
            bundling=BundlingOptions(
                image=RUNTIME.bundling_image,
                local=_UvBundling(self),
                command=[
                    "bash",
                    "-c",
                    f"pip install --quiet --target /asset-output --only-binary :all: {requirements}"
                    + (f" && rm -rf {prune}" if prune else "")
                    + f" && cp -r {copy} /asset-output/",
                ],
            ),
        )


@jsii.implements(ILocalBundling)
class _UvBundling:
    def __init__(self, bundle: Bundle) -> None:
        self.bundle = bundle

    def try_bundle(self, output_dir: str, _options) -> bool:
        uv = shutil.which("uv")
        if uv is None:
            return False
        self.bundle.build(Path(output_dir), uv)
        return True


ROUND = Bundle(
    packages=("common", "db", "lambdas", "selection"),
    libraries=("numpy", "psycopg", "psycopg-binary", "pyarrow", "pydantic", "pyyaml"),
    # Lambda unzips code into at most 250 MB, and pyarrow alone is ~150 MB:
    # drop Arrow Flight, C headers and test suites.
    prune=(
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
    ),
)
API = Bundle(
    packages=("api",),
    libraries=("fastapi", "mangum", "pydantic", "starlette"),
    prune=("bin",),
)
