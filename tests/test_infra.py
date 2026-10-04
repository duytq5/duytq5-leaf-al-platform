import pytest

pytest.importorskip("aws_cdk")


def test_app_synthesizes():
    from infra.app import build_app

    assembly = build_app().synth()
    assert assembly.get_stack_by_name("LeafAlPlatform").template is not None
