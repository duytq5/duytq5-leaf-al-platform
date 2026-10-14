"""On EC2 every container logs to CloudWatch (/leaf-al/host) through Docker's
awslogs driver. The override file is used only there (write-env.sh sets
COMPOSE_FILE), so CI and local runs keep Docker's default logging."""

from pathlib import Path

import yaml

EC2 = Path(__file__).resolve().parents[1] / "deploy" / "ec2"


def test_every_service_logs_to_the_host_group():
    services = yaml.safe_load((EC2 / "docker-compose.yml").read_text())["services"]
    override = yaml.safe_load((EC2 / "docker-compose.logs.yml").read_text())["services"]
    assert set(override) == set(services)
    for name, service in override.items():
        logging = service["logging"]
        assert logging["driver"] == "awslogs", name
        options = logging["options"]
        assert options["awslogs-group"] == "/leaf-al/host"
        assert options["awslogs-region"] == "${AWS_REGION}"
        assert options["tag"] == "{{.Name}}"  # one stream per container
        # A CloudWatch outage must not block or stop Postgres.
        assert options["mode"] == "non-blocking"


def test_write_env_turns_it_on():
    assert (
        "COMPOSE_FILE=docker-compose.yml:docker-compose.logs.yml"
        in (EC2 / "write-env.sh").read_text()
    )
