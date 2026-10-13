import json

from lambdas.post_confirmation.handler import UPLOADERS_GROUP, handle


class FakeCognito:
    def __init__(self):
        self.calls = []

    def admin_add_user_to_group(self, **kwargs):
        self.calls.append(kwargs)


def _event(trigger="PostConfirmation_ConfirmSignUp", identities=True) -> dict:
    attributes = {"sub": "abc", "email": "user@example.com"}
    if identities:
        attributes["identities"] = json.dumps(
            [{"providerName": "Google", "providerType": "Google", "userId": "123"}]
        )
    return {
        "version": "1",
        "triggerSource": trigger,
        "region": "ap-southeast-2",
        "userPoolId": "ap-southeast-2_TEST",
        "userName": "google_123",
        "request": {"userAttributes": attributes},
        "response": {},
    }


def test_google_user_joins_uploaders():
    cognito = FakeCognito()
    event = _event()
    assert handle(event, cognito) is event  # Cognito needs the event back
    assert cognito.calls == [
        {"UserPoolId": "ap-southeast-2_TEST", "Username": "google_123", "GroupName": "uploaders"}
    ]
    assert UPLOADERS_GROUP == "uploaders"


def test_retry_is_harmless():
    # AdminAddUserToGroup is idempotent, so a retried trigger just repeats it.
    cognito = FakeCognito()
    handle(_event(), cognito)
    handle(_event(), cognito)
    assert len(cognito.calls) == 2
    assert cognito.calls[0] == cognito.calls[1]


def test_owner_created_user_gets_no_group():
    # Admins and labelers are created by the owner; their groups are set by hand.
    cognito = FakeCognito()
    handle(_event(identities=False), cognito)
    assert cognito.calls == []


def test_forgot_password_confirmation_is_ignored():
    cognito = FakeCognito()
    handle(_event(trigger="PostConfirmation_ConfirmForgotPassword"), cognito)
    assert cognito.calls == []
