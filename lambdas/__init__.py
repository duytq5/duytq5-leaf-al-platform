"""AWS Lambda handlers.

post_confirmation  Cognito trigger, puts Google sign-ins into `uploaders`
round              the steps of the al-round state machine (Postgres + S3)
labeling           the label step and the webhook's Postgres side (Label Studio)
label_studio_hook  POST /v1/hooks/label-studio, outside the VPC: resumes rounds
upload             app images: incoming/ -> raw/ and the train pool (SQS)
Later: release steps.

Handlers that use Postgres run inside the VPC (lambdas/postgres.py). Step
Functions and Cognito retry them, so each must be safe to run twice.
"""
