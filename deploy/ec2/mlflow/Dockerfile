FROM python:3.11-slim

# SQLAlchemy 2.1+ (pulled in by MLflow 3) maps postgresql:// URIs to psycopg 3,
# not psycopg2, so install psycopg 3.
RUN pip install --no-cache-dir "mlflow>=3.0,<4" "psycopg[binary]>=3.1" boto3

EXPOSE 5000
