import aws_cdk as cdk

from infra.stacks.platform import PlatformStack


def build_app(context: dict | None = None) -> cdk.App:
    app = cdk.App(context=context)
    # Environment-agnostic on purpose: synth needs no AWS lookups, so it runs
    # offline and gives the same template everywhere. `cdk deploy` targets the
    # account and region of your AWS profile.
    PlatformStack(app, "LeafAlPlatform")
    return app


if __name__ == "__main__":
    build_app().synth()
