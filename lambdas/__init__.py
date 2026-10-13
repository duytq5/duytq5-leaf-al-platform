"""AWS Lambda handlers, one subpackage per function.

post_confirmation: Cognito trigger, puts Google sign-ins into `uploaders`.
Later: select (runs a registered strategy), edge API, upload validation,
Label Studio webhook.
Handlers run outside a VPC and must be safe to retry.
"""
