import os

import aws_cdk as cdk

from infra.stacks.platform import PlatformStack


def build_app() -> cdk.App:
    app = cdk.App()
    PlatformStack(
        app,
        "LeafAlPlatform",
        env=cdk.Environment(
            account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
            region=os.environ.get("CDK_DEFAULT_REGION"),
        ),
    )
    return app


if __name__ == "__main__":
    build_app().synth()
