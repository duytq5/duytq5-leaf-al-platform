"""Cognito post-confirmation trigger: put Google sign-ins into `uploaders`.

App users sign in with Google the first time they send an image ("sign in to
contribute"). Cognito creates a federated user and runs this trigger once; the
user lands in `uploaders` so `POST /v1/captures` accepts them. Accounts the
owner creates (admins, labelers) have no `identities` attribute and are left
alone: the owner puts them in groups by hand.

Lambda packages this directory on its own, so the handler imports nothing from
the repo. AdminAddUserToGroup is idempotent, so a retried trigger is harmless.
"""

import boto3

UPLOADERS_GROUP = "uploaders"
SIGN_UP = "PostConfirmation_ConfirmSignUp"

_client = None


def handle(event: dict, cognito) -> dict:
    attributes = event["request"]["userAttributes"]
    if event["triggerSource"] == SIGN_UP and "identities" in attributes:
        cognito.admin_add_user_to_group(
            UserPoolId=event["userPoolId"],
            Username=event["userName"],
            GroupName=UPLOADERS_GROUP,
        )
    return event


def handler(event: dict, context) -> dict:
    global _client
    if _client is None:
        _client = boto3.client("cognito-idp")
    return handle(event, _client)
