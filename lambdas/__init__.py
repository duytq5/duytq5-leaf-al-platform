"""AWS Lambda handlers.

post_confirmation  Cognito trigger, puts Google sign-ins into `uploaders`
round              the steps of the al-round state machine (Postgres + S3)
Later: edge API, upload validation, Label Studio webhook, release steps.

Handlers that use Postgres run inside the VPC (lambdas/postgres.py). Step
Functions and Cognito retry them, so each must be safe to run twice.
"""
