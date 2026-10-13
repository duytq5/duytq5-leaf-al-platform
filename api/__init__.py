"""Edge and Admin API: FastAPI on Lambda behind API Gateway (HTTP API).

API Gateway's Cognito JWT authorizer verifies the access token before the
Lambda runs; this package reads the verified claims and checks groups
(`api.auth.require_groups`). Errors use the shape in `api.errors`.
"""
