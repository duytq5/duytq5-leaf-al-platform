"""AWS Lambda handlers, one subpackage per function.

Phase 1: select (runs a registered strategy), oracle (simulation labeler).
Phase 2: edge API, upload validation, Label Studio webhook.
Handlers run outside a VPC and must be safe to retry.
"""
