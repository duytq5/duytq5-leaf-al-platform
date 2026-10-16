"""The single CDK stack for the POC.

This stack holds the core infrastructure: S3 buckets (al-data, al-models), the
SQS job queue + DLQ, the worker's IAM user, the EC2 host with its retained data
volume, SSM parameters for non-secret settings, the Budget alert, and the
Cognito user pool that signs people in, the al-round state machine with
its Lambdas (infra/stacks/al_round.py), the HTTP API, and Label Studio's
public entry and webhook.

The EC2 host has no inbound ports except HTTPS (80 and 443, Caddy, then an
oauth2-proxy Cognito gate): Label Studio for `labelers` / `admins`, the image
viewer and MLflow for `admins`. The owner and the worker also reach MLflow
(5000), and Postgres (5432), through SSM Session Manager port forwarding, so
that access depends on IAM, not on an IP address. The host keeps one Elastic
IP across stop and start, so the public addresses (and the Cognito callback
URLs) never change.

Lambdas that use Postgres run in a private subnet of the same VPC and reach
the host on its private IP; the host's security group lets in 5432 from them
only. They have no internet route (no NAT gateway): S3 is reached through a
free gateway endpoint, and the Postgres password and CA are put into their
settings by CloudFormation at deploy time.

Deploy-time settings come from CDK context (`cdk deploy -c key=value`):
  budgetEmail   where the Budget alert goes; no Budget is created without it
  budgetUsd     monthly budget in USD (default 20)
  instanceType  EC2 instance type (default t3.medium)
  dataVolumeGb  size of the /data EBS volume (default 20)
  authDomainPrefix    prefix of the Cognito login domain (default leaf-al-<account id>)
  googleClientId      Google OAuth client id; no Google sign-in without it
  mobileCallbackUrls  comma-separated sign-in redirects of the app (default leafal://auth/callback)
  adminCallbackUrls   comma-separated sign-in redirects of admin tools
                      (default http://localhost:8080/auth/callback)
  uploadDailyLimit    images one app user may send in any 24 hours (default 100)
  labelStudioDomain   Label Studio's host name (default <Elastic IP>.sslip.io,
                      a free name that resolves to the IP, so Caddy can get a
                      Let's Encrypt certificate without owning a domain)
  viewerDomain        image viewer's (FiftyOne) host name (default
                      view.<Elastic IP>.sslip.io)
  mlflowDomain        MLflow's host name (default mlflow.<Elastic IP>.sslip.io)
"""

from pathlib import Path

from aws_cdk import (
    Acknowledgment,
    Annotations,
    ArnFormat,
    Aws,
    CfnOutput,
    Duration,
    RemovalPolicy,
    SecretValue,
    Size,
    Stack,
    Tags,
    Validations,
)
from aws_cdk import aws_apigatewayv2 as apigw
from aws_cdk import aws_apigatewayv2_authorizers as apigw_auth
from aws_cdk import aws_apigatewayv2_integrations as apigw_int
from aws_cdk import aws_budgets as budgets
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_lambda_event_sources as lambda_events
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_notifications as s3n
from aws_cdk import aws_sqs as sqs
from aws_cdk import aws_ssm as ssm
from constructs import Construct

from infra import lambda_code
from infra.logs import kebab, lambda_logging, log_group, saved_queries
from infra.stacks.al_round import STATE_MACHINE_NAME as ROUND_STATE_MACHINE
from infra.stacks.al_round import AlRound
from infra.stacks.al_train import STATE_MACHINE_NAME as TRAIN_STATE_MACHINE
from infra.stacks.al_train import AlTrain
from infra.stacks.model_release import STATE_MACHINE_NAME as RELEASE_STATE_MACHINE
from infra.stacks.model_release import ModelRelease

# SSM Parameter Store layout. deploy/ec2/write-env.sh reads the same names.
SSM_PREFIX = "/leaf-al"
SSM_DATA_BUCKET = f"{SSM_PREFIX}/buckets/data"
SSM_MODELS_BUCKET = f"{SSM_PREFIX}/buckets/models"
SSM_JOB_QUEUE_URL = f"{SSM_PREFIX}/queues/jobs-url"
# SecureString: CloudFormation cannot create these, the owner does (docs/aws-setup.md).
SSM_POSTGRES_PASSWORD = f"{SSM_PREFIX}/postgres/password"
SSM_POSTGRES_TLS_CA = f"{SSM_PREFIX}/postgres/tls-ca"
# Secrets Manager copy of the Postgres password for the Lambdas, created by the
# owner (docs/aws-setup.md). CloudFormation resolves it at deploy time, so it is
# never in the template; a Lambda without internet access could not fetch it.
POSTGRES_PASSWORD_SECRET = "leaf-al/postgres-password"
SSM_AL_ROUND_ARN = f"{SSM_PREFIX}/state-machines/al-round"
SSM_MODEL_RELEASE_ARN = f"{SSM_PREFIX}/state-machines/model-release"
SSM_AL_TRAIN_ARN = f"{SSM_PREFIX}/state-machines/al-train"
# Label Studio settings write-env.sh reads on the host.
SSM_USER_POOL_ID = f"{SSM_PREFIX}/auth/user-pool-id"
SSM_LABEL_STUDIO_DOMAIN = f"{SSM_PREFIX}/label-studio/domain"
SSM_LABEL_STUDIO_CLIENT_ID = f"{SSM_PREFIX}/label-studio/client-id"
SSM_VIEWER_DOMAIN = f"{SSM_PREFIX}/viewer/domain"
SSM_MLFLOW_DOMAIN = f"{SSM_PREFIX}/mlflow/domain"
# Secrets Manager JSON created by the owner (docs/aws-setup.md): admin_email,
# admin_password, api_token, webhook_secret, cookie_secret. The host reads it
# at setup; CloudFormation puts api_token and webhook_secret into the Lambdas.
LABEL_STUDIO_SECRET = "leaf-al/label-studio"
LABEL_STUDIO_PORT = 8080
WEBHOOK_PATH = "/v1/hooks/label-studio"

MLFLOW_PORT = 5000
POSTGRES_PORT = 5432
# AWS-managed Session Manager document for local port forwarding.
PORT_FORWARDING_DOCUMENT = "AWS-StartPortForwardingSession"

# Long enough for the worker to pick a job up; it extends visibility while a
# job runs (ChangeMessageVisibility), so a crashed job reappears in 30 minutes.
JOB_VISIBILITY_TIMEOUT = Duration.minutes(30)
# Jobs wait in the queue while the GPU machine is off.
JOB_RETENTION = Duration.days(14)
JOB_MAX_RECEIVES = 3
# App uploads (incoming/) wait in a queue for the validation Lambda. While the
# EC2 host is stopped Postgres is unreachable: a message comes back every
# 15 minutes for about a day, then waits two weeks in the DLQ for a redrive.
UPLOAD_VISIBILITY_TIMEOUT = Duration.minutes(15)
UPLOAD_MAX_RECEIVES = 100
UPLOAD_RETENTION = Duration.days(14)

AL2023_AMI_PARAM = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
DOCKER_COMPOSE_VERSION = "v2.29.7"
DOCKER_BUILDX_VERSION = "v0.17.1"
GITHUB_REPO = "duytq5/duytq5-leaf-al-platform"
REPO_URL = f"https://github.com/{GITHUB_REPO}.git"
# The repo with configs/rounds/ and configs/releases/ and the workflows that
# act on them (duytq5/active-learning-operation), by GitHub's immutable OIDC
# subject: owner and repo ids next to the names, so a new repo that takes the
# name later cannot get the role. Read it with
# `gh api repos/<owner>/<name>/actions/oidc/customization/sub` (sub_claim_prefix).
# Another ops repo: `-c opsRepoSub=<its prefix>`, or `-c opsRepo=owner/name`
# for a repo on the classic, name-only subject.
OPS_REPO_SUB = "repo:duytq5@88496947/active-learning-operation@1418488226"
GITHUB_OIDC_HOST = "token.actions.githubusercontent.com"
GITHUB_ROLE_NAME = "leaf-al-github-actions"
REPO_DIR = "/home/ec2-user/duytq5-leaf-al-platform"
# The host's subnet already spans the VPC's 10.0.0.0/16, so the Lambdas' private
# subnet sits on a second range; the host and its subnet stay as they are.
LAMBDA_CIDR = "10.1.0.0/24"

REPO_ROOT = Path(__file__).resolve().parents[2]
LAMBDAS_DIR = REPO_ROOT / "lambdas"
# GET routes of the app that need no sign-in (design doc, Edge API).
PUBLIC_ROUTES = ("/v1/crops", "/v1/devices/config")
DEFAULT_UPLOAD_DAILY_LIMIT = 100
# Per-client limits on the whole API (requests per second, burst).
API_RATE_LIMIT = 10
API_BURST_LIMIT = 20

# Cognito groups (design doc, Authentication and authorization). Operator and
# ML engineer are GitHub roles (CODEOWNERS), not Cognito groups.
AUTH_GROUPS = ("admins", "labelers", "uploaders")
AUTH_GROUP_DESCRIPTIONS = {
    "admins": "Read-only Admin API, Label Studio gate; created by the owner",
    "labelers": "Active rounds, Label Studio gate; created by the owner",
    "uploaders": "Send images from the app; added on first Google sign-in",
}
# Secrets Manager secret holding the Google OAuth client secret. Created by the
# owner (docs/aws-setup.md); CloudFormation resolves it at deploy time, so it is
# never in the template.
GOOGLE_CLIENT_SECRET_NAME = "leaf-al/google-client-secret"
DEFAULT_MOBILE_CALLBACK = "leafal://auth/callback"
DEFAULT_ADMIN_CALLBACK = "http://localhost:8080/auth/callback"


class PlatformStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        Tags.of(self).add("project", "leaf-al-platform")

        ctx = self.node.try_get_context
        for removed in ("adminCidr", "workerCidr"):
            if ctx(removed):
                Annotations.of(self).add_warning_v2(
                    f"leaf-al:{removed}-ignored",
                    f"{removed} is ignored: the host has no inbound ports. "
                    "Use SSM port forwarding (docs/aws-setup.md).",
                )

        self.data_bucket = self._bucket("DataBucket", versioned=False)
        self.models_bucket = self._bucket("ModelsBucket", versioned=True)
        self.data_bucket.add_lifecycle_rule(
            id="expire-old-postgres-backups",
            prefix="backups/",
            expiration=Duration.days(30),
        )

        self.job_dlq = sqs.Queue(
            self,
            "JobDlq",
            retention_period=JOB_RETENTION,
            encryption=sqs.QueueEncryption.SQS_MANAGED,
            enforce_ssl=True,
        )
        self.job_queue = sqs.Queue(
            self,
            "JobQueue",
            visibility_timeout=JOB_VISIBILITY_TIMEOUT,
            retention_period=JOB_RETENTION,
            receive_message_wait_time=Duration.seconds(20),
            encryption=sqs.QueueEncryption.SQS_MANAGED,
            enforce_ssl=True,
            dead_letter_queue=sqs.DeadLetterQueue(
                queue=self.job_dlq, max_receive_count=JOB_MAX_RECEIVES
            ),
        )

        self._parameters()
        self.worker_user = self._worker_user()
        self._ec2_host()
        self._lambda_network()
        self._al_round()
        self._al_train()
        self._round_control()
        self._model_release()
        self._github_actions()
        self._budget(ctx("budgetEmail"), float(ctx("budgetUsd") or 20))
        self._auth(ctx)
        self._api()
        self._label_studio()
        self._upload_validation()
        saved_queries(self)

        CfnOutput(self, "DataBucketName", value=self.data_bucket.bucket_name)
        CfnOutput(self, "ModelsBucketName", value=self.models_bucket.bucket_name)
        CfnOutput(self, "JobQueueUrl", value=self.job_queue.queue_url)
        CfnOutput(self, "JobDlqUrl", value=self.job_dlq.queue_url)
        CfnOutput(self, "WorkerUserName", value=self.worker_user.user_name)

    def _bucket(self, construct_id: str, *, versioned: bool) -> s3.Bucket:
        # Names are generated by CloudFormation (S3 names are global); the
        # stack writes them to SSM. Retained on `cdk destroy`: data is not
        # thrown away with the stack.
        rules = [
            s3.LifecycleRule(
                id="abort-incomplete-uploads",
                abort_incomplete_multipart_upload_after=Duration.days(7),
            )
        ]
        if versioned:
            rules.append(
                s3.LifecycleRule(
                    id="expire-old-versions", noncurrent_version_expiration=Duration.days(90)
                )
            )
        return s3.Bucket(
            self,
            construct_id,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            object_ownership=s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=versioned,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=rules,
        )

    def _parameters(self) -> None:
        for construct_id, name, value in [
            ("DataBucketParam", SSM_DATA_BUCKET, self.data_bucket.bucket_name),
            ("ModelsBucketParam", SSM_MODELS_BUCKET, self.models_bucket.bucket_name),
            ("JobQueueUrlParam", SSM_JOB_QUEUE_URL, self.job_queue.queue_url),
        ]:
            ssm.StringParameter(self, construct_id, parameter_name=name, string_value=value)

    def _param_arn(self, name: str) -> str:
        return self.format_arn(service="ssm", resource="parameter", resource_name=name.lstrip("/"))

    def _worker_user(self) -> iam.User:
        """IAM user for the local RTX 5070 worker. The access key is created by
        the owner (docs/aws-setup.md), never by CloudFormation, so it does not
        end up in stack outputs or templates."""
        user = iam.User(self, "WorkerUser")
        # A managed policy, not the user's inline policy: IAM caps a user's
        # inline policies at 2,048 characters in total, a managed policy at 6,144.
        self.worker_policy = iam.ManagedPolicy(self, "WorkerPolicy", users=[user])
        policy = self.worker_policy
        data, models = self.data_bucket, self.models_bucket

        # The agent's log lines and job output (worker/logs.py).
        policy.add_statements(_write_logs(log_group(self, "WorkerLogs", "worker")))
        policy.add_statements(
            iam.PolicyStatement(
                sid="ConsumeJobs",
                actions=[
                    "sqs:ReceiveMessage",
                    "sqs:DeleteMessage",
                    "sqs:ChangeMessageVisibility",
                    "sqs:GetQueueAttributes",
                    "sqs:GetQueueUrl",
                ],
                resources=[self.job_queue.queue_arn],
            )
        )
        # These Step Functions actions do not support resource-level
        # permissions; the task token itself scopes what the worker can resume.
        policy.add_statements(
            iam.PolicyStatement(
                sid="ReportTaskResults",
                actions=[
                    "states:SendTaskSuccess",
                    "states:SendTaskFailure",
                    "states:SendTaskHeartbeat",
                ],
                resources=["*"],
            )
        )
        # train/score read images and manifests, score writes predictions.
        policy.add_statements(
            iam.PolicyStatement(
                sid="ReadTrainingData",
                actions=["s3:GetObject"],
                resources=[data.arn_for_objects("raw/*"), data.arn_for_objects("manifests/*")],
            )
        )
        policy.add_statements(
            iam.PolicyStatement(
                sid="WriteScores",
                actions=["s3:GetObject", "s3:PutObject"],
                resources=[data.arn_for_objects("scores/*")],
            )
        )
        policy.add_statements(
            iam.PolicyStatement(
                sid="ListDataPrefixes",
                actions=["s3:ListBucket"],
                resources=[data.bucket_arn],
                conditions={"StringLike": {"s3:prefix": ["raw/*", "manifests/*", "scores/*"]}},
            )
        )
        # MLflow clients write artifacts straight to S3 (--no-serve-artifacts).
        policy.add_statements(
            iam.PolicyStatement(
                sid="MlflowArtifacts",
                actions=["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                resources=[models.arn_for_objects("mlflow-artifacts/*")],
            )
        )
        # The export job (F5) publishes model.onnx and manifest.json for the app.
        policy.add_statements(
            iam.PolicyStatement(
                sid="PublishModels",
                actions=["s3:GetObject", "s3:PutObject"],
                resources=[models.arn_for_objects("edge/*")],
            )
        )
        policy.add_statements(
            iam.PolicyStatement(
                sid="ListMlflowArtifacts",
                actions=["s3:ListBucket"],
                resources=[models.bucket_arn],
                conditions={"StringLike": {"s3:prefix": ["mlflow-artifacts/*", "edge/*"]}},
            )
        )
        policy.add_statements(
            iam.PolicyStatement(
                sid="ReadSettings",
                actions=["ssm:GetParameter", "ssm:GetParameters"],
                resources=[
                    self._param_arn(SSM_DATA_BUCKET),
                    self._param_arn(SSM_MODELS_BUCKET),
                    self._param_arn(SSM_JOB_QUEUE_URL),
                ],
            )
        )
        return user

    def _ec2_host(self) -> None:
        """One instance running Postgres + MLflow (docker-compose), with all
        state on a separate EBS volume that is kept when the instance or the
        stack is deleted."""
        # Public subnet only, no NAT gateway: the VPC itself costs nothing.
        self.vpc = vpc = ec2.Vpc(
            self,
            "Vpc",
            max_azs=1,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC)
            ],
        )

        # No inbound rules at all, not even SSH. The owner and the worker use
        # SSM Session Manager (shell and port forwarding), which goes out from
        # the SSM agent, so the instance needs only outbound access.
        self.host_sg = sg = ec2.SecurityGroup(
            self,
            "HostSg",
            vpc=vpc,
            description="leaf-al EC2: no inbound, access through SSM Session Manager",
            allow_all_outbound=True,
        )

        role = iam.Role(
            self,
            "HostRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore")
            ],
        )
        # write-env.sh reads the password and settings.
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["ssm:GetParameter", "ssm:GetParameters"],
                resources=[self._param_arn(f"{SSM_PREFIX}/*")],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["kms:Decrypt"],
                resources=["*"],
                conditions={"StringEquals": {"kms:ViaService": f"ssm.{self.region}.amazonaws.com"}},
            )
        )
        # Nightly pg_dumpall (backup-postgres.sh) and restores.
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:PutObject", "s3:GetObject"],
                resources=[self.data_bucket.arn_for_objects("backups/*")],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                resources=[self.models_bucket.arn_for_objects("mlflow-artifacts/*")],
            )
        )
        # Docker's awslogs driver: every container's output
        # (deploy/ec2/docker-compose.logs.yml).
        role.add_to_policy(_write_logs(log_group(self, "HostLogs", "host")))
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[self.data_bucket.bucket_arn, self.models_bucket.bucket_arn],
                conditions={"StringLike": {"s3:prefix": ["backups/*", "mlflow-artifacts/*"]}},
            )
        )

        # The volume must be in the instance's AZ. Take it from the subnet, not
        # the instance, because the instance's user data refers to the volume.
        subnet = vpc.public_subnets[0]
        user_data = ec2.UserData.for_linux()
        self.host = instance = ec2.Instance(
            self,
            "Host",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnets=[subnet]),
            instance_type=ec2.InstanceType(
                self.node.try_get_context("instanceType") or "t3.medium"
            ),
            # Resolved by EC2 at launch, not by CloudFormation at deploy: a new
            # AMI release never replaces the running instance on `cdk deploy`.
            machine_image=ec2.MachineImage.resolve_ssm_parameter_at_launch(AL2023_AMI_PARAM),
            security_group=sg,
            role=role,
            user_data=user_data,
            require_imdsv2=True,
            associate_public_ip_address=True,
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/xvda",
                    volume=ec2.BlockDeviceVolume.ebs(
                        16, volume_type=ec2.EbsDeviceVolumeType.GP3, encrypted=True
                    ),
                )
            ],
        )

        # cfn-lint expects a literal AMI id; resolve:ssm: is valid for RunInstances.
        Validations.of(instance).acknowledge(
            Acknowledgment(
                id="CloudFormation-Validate::E1152",
                reason="resolve:ssm: AMI id is resolved by EC2 at launch",
            )
        )

        volume = ec2.Volume(
            self,
            "DataVolume",
            availability_zone=subnet.availability_zone,
            size=Size.gibibytes(int(self.node.try_get_context("dataVolumeGb") or 20)),
            volume_type=ec2.EbsDeviceVolumeType.GP3,
            encrypted=True,
            removal_policy=RemovalPolicy.RETAIN,
        )
        Tags.of(volume).add("Name", "leaf-al-data")
        ec2.CfnVolumeAttachment(
            self,
            "DataVolumeAttachment",
            instance_id=instance.instance_id,
            volume_id=volume.volume_id,
            device="/dev/sdf",
        )

        user_data.add_commands(*_host_setup_commands(volume.volume_id, self.region))
        self._worker_tunnel(instance)

        # A fixed public address for Label Studio: the auto-assigned one
        # changes on every stop and start.
        eip = ec2.CfnEIP(self, "HostEip", domain="vpc")
        Tags.of(eip).add("Name", "leaf-al-host")
        ec2.CfnEIPAssociation(
            self,
            "HostEipAssociation",
            allocation_id=eip.attr_allocation_id,
            instance_id=instance.instance_id,
        )
        self.label_studio_domain = (
            self.node.try_get_context("labelStudioDomain") or f"{eip.attr_public_ip}.sslip.io"
        )
        # Label Studio shows each task image from a presigned link to raw/ in
        # the data bucket, loaded cross-origin from its own page: the browser
        # refuses the image without this rule. Only GET, only from Label Studio.
        self.data_bucket.add_cors_rule(
            allowed_methods=[s3.HttpMethods.GET, s3.HttpMethods.HEAD],
            allowed_origins=[f"https://{self.label_studio_domain}"],
            allowed_headers=["*"],
            max_age=3600,
        )
        # The image viewer (FiftyOne) on the same host and Caddy.
        self.viewer_domain = (
            self.node.try_get_context("viewerDomain") or f"view.{eip.attr_public_ip}.sslip.io"
        )
        # MLflow for ML engineers, through Caddy and a Cognito gate.
        self.mlflow_domain = (
            self.node.try_get_context("mlflowDomain") or f"mlflow.{eip.attr_public_ip}.sslip.io"
        )
        # Caddy: ACME HTTP challenge and redirect on 80, Label Studio on 443.
        for port in (80, 443):
            sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(port), "Label Studio (Caddy)")

        CfnOutput(self, "InstanceId", value=instance.instance_id)
        CfnOutput(self, "DataVolumeId", value=volume.volume_id)

    def _lambda_network(self) -> None:
        """A private subnet for the Lambdas, with no internet route, and an S3
        gateway endpoint (free)."""
        vpc, public = self.vpc, self.vpc.public_subnets[0]
        cidr = ec2.CfnVPCCidrBlock(self, "LambdaCidr", vpc_id=vpc.vpc_id, cidr_block=LAMBDA_CIDR)
        subnet = ec2.CfnSubnet(
            self,
            "LambdaSubnet",
            vpc_id=vpc.vpc_id,
            cidr_block=LAMBDA_CIDR,
            availability_zone=public.availability_zone,
            map_public_ip_on_launch=False,
        )
        subnet.add_dependency(cidr)
        Tags.of(subnet).add("Name", "leaf-al-lambdas")
        route_table = ec2.CfnRouteTable(self, "LambdaRouteTable", vpc_id=vpc.vpc_id)
        ec2.CfnSubnetRouteTableAssociation(
            self,
            "LambdaRouteTableAssociation",
            subnet_id=subnet.ref,
            route_table_id=route_table.ref,
        )
        ec2.CfnVPCEndpoint(
            self,
            "S3Endpoint",
            vpc_id=vpc.vpc_id,
            service_name=f"com.amazonaws.{self.region}.s3",
            vpc_endpoint_type="Gateway",
            route_table_ids=[route_table.ref],
        )
        self.lambda_subnet = ec2.Subnet.from_subnet_attributes(
            self,
            "LambdaSubnetRef",
            subnet_id=subnet.ref,
            availability_zone=public.availability_zone,
            route_table_id=route_table.ref,
        )
        self.lambda_sg = ec2.SecurityGroup(
            self,
            "LambdaSg",
            vpc=vpc,
            description="leaf-al Lambdas: Postgres on the host, S3 through the gateway endpoint",
            allow_all_outbound=True,
        )
        self.host_sg.add_ingress_rule(
            self.lambda_sg, ec2.Port.tcp(POSTGRES_PORT), "Postgres from the Lambdas (TLS)"
        )
        # The label step calls Label Studio's API directly, past the Cognito gate,
        # with the admin API token; this port is closed to everyone else.
        self.host_sg.add_ingress_rule(
            self.lambda_sg, ec2.Port.tcp(LABEL_STUDIO_PORT), "Label Studio API from the Lambdas"
        )

    def _postgres_environment(self) -> dict[str, str]:
        """Settings of lambdas/postgres.py open_db() for the Lambdas in the VPC."""
        return {
            "PG_HOSTADDR": self.host.instance_private_ip,
            "PG_PASSWORD": SecretValue.secrets_manager(POSTGRES_PASSWORD_SECRET).unsafe_unwrap(),
            "PG_CA": ssm.StringParameter.value_for_string_parameter(self, SSM_POSTGRES_TLS_CA),
        }

    def _al_round(self) -> None:
        al_round = AlRound(
            self,
            "AlRound",
            vpc=self.vpc,
            subnet=self.lambda_subnet,
            security_group=self.lambda_sg,
            environment=self._postgres_environment(),
            label_studio_environment={
                "LS_URL": f"http://{self.host.instance_private_ip}:{LABEL_STUDIO_PORT}",
                "LS_API_TOKEN": self._label_studio_secret("api_token"),
            },
            data_bucket=self.data_bucket,
            job_queue=self.job_queue,
        )
        self.al_round = al_round
        ssm.StringParameter(
            self,
            "AlRoundArnParam",
            parameter_name=SSM_AL_ROUND_ARN,
            string_value=al_round.state_machine.state_machine_arn,
        )
        CfnOutput(self, "AlRoundStateMachineArn", value=al_round.state_machine.state_machine_arn)

    def _al_train(self) -> None:
        """Standalone trainings (design doc change 24)."""
        al_train = AlTrain(
            self,
            "AlTrain",
            vpc=self.vpc,
            subnet=self.lambda_subnet,
            security_group=self.lambda_sg,
            environment=self._postgres_environment(),
            data_bucket=self.data_bucket,
            job_queue=self.job_queue,
        )
        self.al_train = al_train
        arn = al_train.state_machine.state_machine_arn
        ssm.StringParameter(
            self, "AlTrainArnParam", parameter_name=SSM_AL_TRAIN_ARN, string_value=arn
        )
        CfnOutput(self, "AlTrainStateMachineArn", value=arn)

    def _round_control(self) -> None:
        """The Lambdas the start-round, cancel-round and start-training
        workflows invoke (F3, change 24). Fixed names, so the workflows can
        call them."""

        def control(
            construct_id: str, name: str, handler: str, module: str = "control"
        ) -> lambda_.Function:
            return lambda_.Function(
                self,
                construct_id,
                function_name=name,
                runtime=lambda_code.RUNTIME,
                code=lambda_code.ROUND.code(),
                handler=f"lambdas.{module}.{handler}",
                timeout=Duration.seconds(60),
                memory_size=512,
                vpc=self.vpc,
                vpc_subnets=ec2.SubnetSelection(subnets=[self.lambda_subnet]),
                security_groups=[self.lambda_sg],
                environment={
                    **self._postgres_environment(),
                    "DATA_BUCKET": self.data_bucket.bucket_name,
                },
                **lambda_logging(self, f"{construct_id}Logs", f"control/{kebab(construct_id)}"),
            )

        self.start_round_fn = control("StartRoundFn", "leaf-al-start-round", "start_round_handler")
        self.cancel_round_fn = control(
            "CancelRoundFn", "leaf-al-cancel-round", "cancel_round_handler"
        )
        self.start_training_fn = control(
            "StartTrainingFn", "leaf-al-start-training", "start_training_handler", "training"
        )
        # Cancelling puts the round's selected images back in the pool.
        self.data_bucket.grant_read(self.cancel_round_fn, "selections/*")

    def _model_release(self) -> None:
        release = ModelRelease(
            self,
            "ModelRelease",
            vpc=self.vpc,
            subnet=self.lambda_subnet,
            security_group=self.lambda_sg,
            environment=self._postgres_environment(),
            job_queue=self.job_queue,
        )
        self.model_release = release
        arn = release.state_machine.state_machine_arn
        ssm.StringParameter(
            self, "ModelReleaseArnParam", parameter_name=SSM_MODEL_RELEASE_ARN, string_value=arn
        )
        CfnOutput(self, "ModelReleaseStateMachineArn", value=arn)

    def _github_actions(self) -> None:
        """The IAM role GitHub Actions assumes with its OIDC token (design doc,
        "Control through GitHub"): no AWS keys are stored in GitHub, and only
        workflow runs on `main` of the ops repo (round and release files, owner
        decision 2026-10-14) get the role, so a pull request (forks included)
        never runs with AWS access, and nothing in this code repo does.

        An account has one OIDC provider per URL: pass `-c
        githubOidcProviderArn=...` to reuse one made outside this stack.
        """
        ops_repo = self.node.try_get_context("opsRepo")
        # Another ops repo without its subject: the classic name-only subject.
        ops_sub = self.node.try_get_context("opsRepoSub") or (
            f"repo:{ops_repo}" if ops_repo else OPS_REPO_SUB
        )
        existing = self.node.try_get_context("githubOidcProviderArn")
        if existing:
            provider_arn = existing
        else:
            provider = iam.OidcProviderNative(
                self,
                "GithubOidc",
                url=f"https://{GITHUB_OIDC_HOST}",
                client_ids=["sts.amazonaws.com"],
            )
            provider_arn = provider.oidc_provider_arn
        role = iam.Role(
            self,
            "GithubActionsRole",
            role_name=GITHUB_ROLE_NAME,
            assumed_by=iam.WebIdentityPrincipal(
                provider_arn,
                conditions={
                    "StringEquals": {
                        f"{GITHUB_OIDC_HOST}:aud": "sts.amazonaws.com",
                        f"{GITHUB_OIDC_HOST}:sub": f"{ops_sub}:ref:refs/heads/main",
                    }
                },
            ),
            # The release workflow keeps polling while the worker exports.
            max_session_duration=Duration.hours(7),
        )

        def executions(machine_name: str) -> str:
            return self.format_arn(
                service="states",
                resource="execution",
                resource_name=f"{machine_name}:*",
                arn_format=ArnFormat.COLON_RESOURCE_NAME,
            )

        # Releases: start and watch model-release (F5).
        self.model_release.state_machine.grant_start_execution(role)
        # Rounds: start-round inserts the round, the workflow starts al-round
        # (the Lambda has no route to Step Functions); cancel-round stops it.
        self.al_round.state_machine.grant_start_execution(role)
        # Trainings (change 24): start-training inserts it, the workflow starts al-train.
        self.al_train.state_machine.grant_start_execution(role)
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["states:DescribeExecution"],
                resources=[
                    executions(RELEASE_STATE_MACHINE),
                    executions(ROUND_STATE_MACHINE),
                    executions(TRAIN_STATE_MACHINE),
                ],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["states:StopExecution"], resources=[executions(ROUND_STATE_MACHINE)]
            )
        )
        for fn in (self.start_round_fn, self.cancel_round_fn, self.start_training_fn):
            fn.grant_invoke(role)
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["ssm:GetParameter"],
                resources=[
                    self._param_arn(SSM_MODEL_RELEASE_ARN),
                    self._param_arn(SSM_AL_ROUND_ARN),
                    self._param_arn(SSM_AL_TRAIN_ARN),
                ],
            )
        )
        CfnOutput(self, "GithubActionsRoleArn", value=role.role_arn)

    def _worker_tunnel(self, instance: ec2.Instance) -> None:
        """Let the worker open SSM port-forwarding sessions to the host (MLflow,
        Postgres), and nothing else: no shell, no other instance."""
        instance_arn = self.format_arn(
            service="ec2", resource="instance", resource_name=instance.instance_id
        )
        port_forwarding = self.format_arn(
            service="ssm",
            account="",
            resource="document",
            resource_name=PORT_FORWARDING_DOCUMENT,
        )
        self.worker_policy.add_statements(
            iam.PolicyStatement(
                sid="TunnelToHost",
                actions=["ssm:StartSession"],
                resources=[instance_arn, port_forwarding],
                # Without this, StartSession on the instance also allows the
                # default shell document.
                conditions={"BoolIfExists": {"ssm:SessionDocumentAccessCheck": "true"}},
            )
        )
        self.worker_policy.add_statements(
            iam.PolicyStatement(
                sid="ManageOwnSessions",
                actions=["ssm:TerminateSession", "ssm:ResumeSession"],
                resources=[
                    self.format_arn(
                        service="ssm", resource="session", resource_name="${aws:username}-*"
                    )
                ],
            )
        )

    def _auth(self, ctx) -> None:
        """Cognito user pool for people (machines keep IAM).

        The owner creates admins and labelers; only Google sign-ins from the app
        register themselves, and the post-confirmation trigger puts those in
        `uploaders`. Sign-in goes through the hosted login page with
        authorization code + PKCE, so both app clients are public (no secret)
        and no password API is enabled.
        """
        trigger = self._post_confirmation_fn()
        self.user_pool = cognito.UserPool(
            self,
            "UserPool",
            feature_plan=cognito.FeaturePlan.LITE,
            self_sign_up_enabled=False,
            sign_in_aliases=cognito.SignInAliases(email=True),
            auto_verify=cognito.AutoVerifiedAttrs(email=True),
            standard_attributes=cognito.StandardAttributes(
                email=cognito.StandardAttribute(required=True, mutable=True)
            ),
            password_policy=cognito.PasswordPolicy(min_length=12),
            mfa=cognito.Mfa.OPTIONAL,
            mfa_second_factor=cognito.MfaSecondFactor(sms=False, otp=True),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            lambda_triggers=cognito.UserPoolTriggers(post_confirmation=trigger),
            removal_policy=RemovalPolicy.RETAIN,
        )
        for name in AUTH_GROUPS:
            cognito.UserPoolGroup(
                self,
                f"Group{name.capitalize()}",
                user_pool=self.user_pool,
                group_name=name,
                description=AUTH_GROUP_DESCRIPTIONS[name],
            )

        domain = self.user_pool.add_domain(
            "LoginDomain",
            cognito_domain=cognito.CognitoDomainOptions(
                domain_prefix=ctx("authDomainPrefix") or f"leaf-al-{Aws.ACCOUNT_ID}"
            ),
        )

        google = None
        google_client_id = ctx("googleClientId")
        if google_client_id:
            google = cognito.UserPoolIdentityProviderGoogle(
                self,
                "UserPoolGoogle",
                user_pool=self.user_pool,
                client_id=google_client_id,
                client_secret_value=SecretValue.secrets_manager(GOOGLE_CLIENT_SECRET_NAME),
                scopes=["openid", "email", "profile"],
                attribute_mapping=cognito.AttributeMapping(
                    email=cognito.ProviderAttribute.GOOGLE_EMAIL
                ),
            )
        else:
            Annotations.of(self).add_warning_v2(
                "leaf-al:no-google-client-id",
                "googleClientId is not set: the app has no Google sign-in yet (docs/aws-setup.md).",
            )

        def urls(key: str, default: str) -> list[str]:
            return [u.strip() for u in (ctx(key) or default).split(",") if u.strip()]

        self.mobile_client = self._app_client(
            "MobileClient", "mobile", urls("mobileCallbackUrls", DEFAULT_MOBILE_CALLBACK), google
        )
        self.admin_web_client = self._app_client(
            "AdminWebClient", "admin-web", urls("adminCallbackUrls", DEFAULT_ADMIN_CALLBACK), None
        )
        # oauth2-proxy in front of Label Studio, and the ones in front of the
        # image viewer and MLflow: servers, so they keep a secret.
        self.label_studio_client = self._app_client(
            "LabelStudioClient",
            "label-studio",
            [
                f"https://{self.label_studio_domain}/oauth2/callback",
                f"https://{self.viewer_domain}/oauth2/callback",
                f"https://{self.mlflow_domain}/oauth2/callback",
            ],
            None,
            generate_secret=True,
        )
        for construct_id, name, value in [
            ("UserPoolIdParam", SSM_USER_POOL_ID, self.user_pool.user_pool_id),
            (
                "LabelStudioClientIdParam",
                SSM_LABEL_STUDIO_CLIENT_ID,
                self.label_studio_client.user_pool_client_id,
            ),
            ("LabelStudioDomainParam", SSM_LABEL_STUDIO_DOMAIN, self.label_studio_domain),
            ("ViewerDomainParam", SSM_VIEWER_DOMAIN, self.viewer_domain),
            ("MlflowDomainParam", SSM_MLFLOW_DOMAIN, self.mlflow_domain),
        ]:
            ssm.StringParameter(self, construct_id, parameter_name=name, string_value=value)

        CfnOutput(self, "UserPoolId", value=self.user_pool.user_pool_id)
        CfnOutput(self, "MobileClientId", value=self.mobile_client.user_pool_client_id)
        CfnOutput(self, "AdminWebClientId", value=self.admin_web_client.user_pool_client_id)
        CfnOutput(self, "AuthDomainUrl", value=domain.base_url())

    def _app_client(
        self,
        construct_id: str,
        name: str,
        callback_urls: list[str],
        google: cognito.UserPoolIdentityProviderGoogle | None,
        *,
        generate_secret: bool = False,
    ) -> cognito.UserPoolClient:
        providers = [cognito.UserPoolClientIdentityProvider.COGNITO]
        if google:
            providers.append(cognito.UserPoolClientIdentityProvider.GOOGLE)
        client = self.user_pool.add_client(
            construct_id,
            user_pool_client_name=name,
            generate_secret=generate_secret,
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(authorization_code_grant=True),
                scopes=[
                    cognito.OAuthScope.OPENID,
                    cognito.OAuthScope.EMAIL,
                    cognito.OAuthScope.PROFILE,
                ],
                callback_urls=callback_urls,
            ),
            supported_identity_providers=providers,
            prevent_user_existence_errors=True,
            access_token_validity=Duration.hours(1),
            id_token_validity=Duration.hours(1),
            refresh_token_validity=Duration.days(30),
        )
        # Left out, Cognito would also allow SRP and custom auth. Only the hosted
        # login and token refresh are needed.
        cfn_client: cognito.CfnUserPoolClient = client.node.default_child
        cfn_client.explicit_auth_flows = ["ALLOW_REFRESH_TOKEN_AUTH"]
        if google:
            client.node.add_dependency(google)
        return client

    def _api(self) -> None:
        """HTTP API in front of one FastAPI Lambda.

        Routes need a Cognito access token from the `mobile` or `admin-web`
        client: the JWT authorizer checks signature, issuer, client id and
        expiry and answers 401 itself. The Lambda then checks groups
        (`api.auth.require_groups`). The app's catalog (`PUBLIC_ROUTES`) needs
        no token, so every install gets model updates; API Gateway throttles it.

        The Lambda reads and writes Postgres, so it runs in the Lambda subnet;
        it signs S3 download and upload links locally and reads round
        selections through the S3 gateway endpoint; it makes no other AWS call.
        """
        upload_limit = int(
            self.node.try_get_context("uploadDailyLimit") or DEFAULT_UPLOAD_DAILY_LIMIT
        )
        if upload_limit < 1:
            raise ValueError("uploadDailyLimit must be at least 1")
        fn = lambda_.Function(
            self,
            "ApiFn",
            runtime=lambda_code.RUNTIME,
            architecture=lambda_.Architecture.X86_64,
            handler="api.app.handler",
            code=lambda_code.API.code(),
            memory_size=512,
            # Past the 10 s Postgres connect timeout: while the host is stopped
            # the API answers 503 instead of timing out.
            timeout=Duration.seconds(15),
            vpc=self.vpc,
            vpc_subnets=ec2.SubnetSelection(subnets=[self.lambda_subnet]),
            security_groups=[self.lambda_sg],
            environment={
                **self._postgres_environment(),
                "MODELS_BUCKET": self.models_bucket.bucket_name,
                "DATA_BUCKET": self.data_bucket.bucket_name,
                # A deploy setting: the Lambda has no route to SSM.
                "UPLOAD_DAILY_LIMIT": str(upload_limit),
                # Links in the admin API: a round's Label Studio project, and
                # its MLflow run (behind the Cognito gate).
                "LABEL_STUDIO_URL": f"https://{self.label_studio_domain}",
                "MLFLOW_URL": f"https://{self.mlflow_domain}",
            },
            **lambda_logging(self, "ApiLogs", "api"),
        )
        # Presigned links work only for what the signer may do: download
        # released models, upload app images to incoming/. A round's
        # selection file gives its phase (GET /v1/rounds/...). Without
        # ListBucket a missing file is AccessDenied, not NoSuchKey; S3 checks it
        # on the bucket with no prefix, so it cannot be narrowed to selections/
        # (key names only; the Lambda reads them from Postgres anyway).
        fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[
                    self.models_bucket.arn_for_objects("edge/*"),
                    self.data_bucket.arn_for_objects("selections/*"),
                ],
            )
        )
        fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[self.data_bucket.bucket_arn],
            )
        )
        fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:PutObject"], resources=[self.data_bucket.arn_for_objects("incoming/*")]
            )
        )
        authorizer = apigw_auth.HttpUserPoolAuthorizer(
            "CognitoAuthorizer",
            self.user_pool,
            user_pool_clients=[self.mobile_client, self.admin_web_client],
        )
        integration = apigw_int.HttpLambdaIntegration("ApiIntegration", fn)
        self.http_api = api = apigw.HttpApi(self, "Api", default_authorizer=authorizer)
        api.add_routes(path="/v1/{proxy+}", methods=[apigw.HttpMethod.ANY], integration=integration)
        for path in PUBLIC_ROUTES:
            api.add_routes(
                path=path,
                methods=[apigw.HttpMethod.GET],
                integration=integration,
                authorizer=apigw.HttpNoneAuthorizer(),
            )
        stage: apigw.CfnStage = api.default_stage.node.default_child
        stage.default_route_settings = apigw.CfnStage.RouteSettingsProperty(
            throttling_rate_limit=API_RATE_LIMIT, throttling_burst_limit=API_BURST_LIMIT
        )
        CfnOutput(self, "ApiUrl", value=api.api_endpoint)

    def _label_studio_secret(self, field: str) -> str:
        """A field of the owner-made Label Studio secret, resolved by
        CloudFormation at deploy time (the VPC Lambdas cannot fetch it)."""
        return SecretValue.secrets_manager(LABEL_STUDIO_SECRET, json_field=field).unsafe_unwrap()

    def _label_studio(self) -> None:
        """What Label Studio on the host needs from AWS, and its webhook.

        The webhook Lambda runs outside the VPC (it calls SendTaskSuccess; the
        VPC has no NAT) and invokes RecordLabelFn inside the VPC for Postgres.
        Its route has no JWT authorizer: Label Studio sends the shared secret.
        """
        webhook_secret = self._label_studio_secret("webhook_secret")
        record_fn = self.al_round.record_label_fn
        hook = lambda_.Function(
            self,
            "LabelStudioHookFn",
            runtime=lambda_code.RUNTIME,
            code=lambda_code.ROUND.code(),
            handler="lambdas.label_studio_hook.handler",
            timeout=Duration.seconds(30),
            memory_size=256,
            environment={
                "RECORD_LABEL_FUNCTION": record_fn.function_name,
                "LS_WEBHOOK_SECRET": webhook_secret,
            },
            **lambda_logging(self, "LabelStudioHookLogs", "label-studio-hook"),
        )
        record_fn.grant_invoke(hook)
        # SendTaskSuccess has no resource-level permissions; the token scopes it.
        hook.add_to_role_policy(
            iam.PolicyStatement(actions=["states:SendTaskSuccess"], resources=["*"])
        )
        self.http_api.add_routes(
            path=WEBHOOK_PATH,
            methods=[apigw.HttpMethod.POST],
            integration=apigw_int.HttpLambdaIntegration("LabelStudioHookIntegration", hook),
            authorizer=apigw.HttpNoneAuthorizer(),
        )

        start_fn = self.al_round.start_labeling_fn
        start_fn.add_environment("LS_WEBHOOK_URL", f"{self.http_api.api_endpoint}{WEBHOOK_PATH}")
        start_fn.add_environment("LS_WEBHOOK_SECRET", webhook_secret)

        # write-env.sh reads the secret and the gate's client secret; Label
        # Studio signs links to raw/ images with the instance role. A separate
        # policy, so the instance itself does not depend on Cognito.
        data = self.data_bucket
        iam.Policy(
            self,
            "HostLabelStudioPolicy",
            roles=[self.host.role],
            statements=[
                iam.PolicyStatement(
                    actions=["secretsmanager:GetSecretValue"],
                    resources=[
                        self.format_arn(
                            service="secretsmanager",
                            resource="secret",
                            resource_name=f"{LABEL_STUDIO_SECRET}-*",
                            arn_format=ArnFormat.COLON_RESOURCE_NAME,
                        )
                    ],
                ),
                iam.PolicyStatement(
                    actions=["cognito-idp:DescribeUserPoolClient"],
                    resources=[self.user_pool.user_pool_arn],
                ),
                iam.PolicyStatement(
                    actions=["s3:GetObject"], resources=[data.arn_for_objects("raw/*")]
                ),
                iam.PolicyStatement(
                    actions=["s3:ListBucket"],
                    resources=[data.bucket_arn],
                    conditions={"StringLike": {"s3:prefix": ["raw/*"]}},
                ),
            ],
        )
        CfnOutput(self, "LabelStudioUrl", value=f"https://{self.label_studio_domain}")
        CfnOutput(self, "ViewerUrl", value=f"https://{self.viewer_domain}")
        CfnOutput(self, "MlflowUrl", value=f"https://{self.mlflow_domain}")

    def _upload_validation(self) -> None:
        """App images: S3 incoming/ -> UploadQueue -> UploadFn (design doc F1
        step 6, lambdas/upload.py), which moves good images to raw/ and the
        train pool. In the VPC for Postgres; S3 through the gateway endpoint."""
        data = self.data_bucket
        dlq = sqs.Queue(
            self,
            "UploadDlq",
            retention_period=UPLOAD_RETENTION,
            encryption=sqs.QueueEncryption.SQS_MANAGED,
            enforce_ssl=True,
        )
        queue = sqs.Queue(
            self,
            "UploadQueue",
            visibility_timeout=UPLOAD_VISIBILITY_TIMEOUT,
            retention_period=UPLOAD_RETENTION,
            encryption=sqs.QueueEncryption.SQS_MANAGED,
            enforce_ssl=True,
            dead_letter_queue=sqs.DeadLetterQueue(queue=dlq, max_receive_count=UPLOAD_MAX_RECEIVES),
        )
        data.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.SqsDestination(queue),
            s3.NotificationKeyFilter(prefix="incoming/"),
        )
        # Anything never processed (the DLQ keeps messages 14 days) goes away.
        data.add_lifecycle_rule(
            id="expire-unprocessed-uploads", prefix="incoming/", expiration=UPLOAD_RETENTION
        )
        fn = lambda_.Function(
            self,
            "UploadFn",
            runtime=lambda_code.RUNTIME,
            code=lambda_code.UPLOAD.code(),
            handler="lambdas.upload.handler",
            timeout=Duration.seconds(60),
            memory_size=1024,
            vpc=self.vpc,
            vpc_subnets=ec2.SubnetSelection(subnets=[self.lambda_subnet]),
            security_groups=[self.lambda_sg],
            environment={**self._postgres_environment(), "DATA_BUCKET": data.bucket_name},
            **lambda_logging(self, "UploadLogs", "upload"),
        )
        fn.add_event_source(
            lambda_events.SqsEventSource(queue, batch_size=5, report_batch_item_failures=True)
        )
        fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:DeleteObject"],
                resources=[data.arn_for_objects("incoming/*")],
            )
        )
        fn.add_to_role_policy(
            iam.PolicyStatement(actions=["s3:PutObject"], resources=[data.arn_for_objects("raw/*")])
        )
        CfnOutput(self, "UploadDlqUrl", value=dlq.queue_url)

    def _post_confirmation_fn(self) -> lambda_.Function:
        fn = lambda_.Function(
            self,
            "PostConfirmationFn",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=lambda_.Code.from_asset(
                str(LAMBDAS_DIR / "post_confirmation"), exclude=["__pycache__", "*.pyc"]
            ),
            timeout=Duration.seconds(10),
            **lambda_logging(self, "PostConfirmationLogs", "post-confirmation"),
        )
        # The pool ARN would make a cycle (the pool references this function),
        # so the grant covers user pools of this account and region; the
        # function only ever acts on the pool that invoked it.
        fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["cognito-idp:AdminAddUserToGroup"],
                resources=[
                    self.format_arn(service="cognito-idp", resource="userpool", resource_name="*")
                ],
            )
        )
        return fn

    def _budget(self, email: str | None, amount_usd: float) -> None:
        if not email:
            Annotations.of(self).add_warning_v2(
                "leaf-al:no-budget-email",
                "budgetEmail not set: no Budget alert is created. "
                "Deploy with -c budgetEmail=<you@example.com>.",
            )
            return
        subscribers = [
            budgets.CfnBudget.SubscriberProperty(subscription_type="EMAIL", address=email)
        ]
        budgets.CfnBudget(
            self,
            "MonthlyBudget",
            budget=budgets.CfnBudget.BudgetDataProperty(
                budget_name="leaf-al-monthly",
                budget_type="COST",
                time_unit="MONTHLY",
                budget_limit=budgets.CfnBudget.SpendProperty(amount=amount_usd, unit="USD"),
            ),
            notifications_with_subscribers=[
                budgets.CfnBudget.NotificationWithSubscribersProperty(
                    notification=budgets.CfnBudget.NotificationProperty(
                        notification_type=kind,
                        comparison_operator="GREATER_THAN",
                        threshold=threshold,
                        threshold_type="PERCENTAGE",
                    ),
                    subscribers=subscribers,
                )
                for kind, threshold in [("ACTUAL", 80), ("FORECASTED", 100)]
            ],
        )


def _host_setup_commands(volume_id: str, region: str) -> list[str]:
    """First-boot setup: Docker + compose, cron, 2 GB swap, /data on the retained volume,
    then the app: clone the repo and run deploy/ec2/bootstrap.sh (Postgres +
    MLflow + nightly backup). The Postgres password must already be in SSM
    (docs/aws-setup.md section 1); if it is not, this fails and is logged in
    /var/log/cloud-init-output.log. Run `scripts/ec2.sh setup` to retry.

    The volume is found by its id (Nitro exposes it as an NVMe device whose
    serial is the id), so the device name does not matter. It is formatted
    only if it has no filesystem yet: a re-attached volume keeps its data.
    """
    arch = "x86_64"
    plugins = "/usr/local/lib/docker/cli-plugins"
    gh = "https://github.com/docker"
    return [
        "set -euxo pipefail",
        "dnf install -y docker git cronie",
        f"mkdir -p {plugins}",
        f"curl -fsSL {gh}/compose/releases/download/{DOCKER_COMPOSE_VERSION}/"
        f"docker-compose-linux-{arch} -o {plugins}/docker-compose",
        f"curl -fsSL {gh}/buildx/releases/download/{DOCKER_BUILDX_VERSION}/"
        f"buildx-{DOCKER_BUILDX_VERSION}.linux-amd64 -o {plugins}/docker-buildx",
        f"chmod +x {plugins}/docker-compose {plugins}/docker-buildx",
        # Swap as a safety net: on t3.small (2 GB) a crash-looping MLflow next to
        # Postgres starved the host until the SSM agent stopped answering.
        "if [ ! -f /swapfile ]; then dd if=/dev/zero of=/swapfile bs=1M count=2048 "
        "&& chmod 600 /swapfile && mkswap /swapfile; fi",
        "swapon /swapfile || true",
        "grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab",
        "systemctl enable --now docker crond",
        "usermod -aG docker ec2-user",
        f"VOL={volume_id}",
        'DEV="/dev/disk/by-id/nvme-Amazon_Elastic_Block_Store_${VOL/-/}"',
        'for i in $(seq 1 60); do [ -e "$DEV" ] && break; sleep 5; done',
        '[ -e "$DEV" ]',
        'if ! blkid "$DEV"; then mkfs.ext4 -L leaf-al-data "$DEV"; fi',
        "mkdir -p /data",
        "grep -q 'LABEL=leaf-al-data' /etc/fstab || "
        "echo 'LABEL=leaf-al-data /data ext4 defaults,nofail 0 2' >> /etc/fstab",
        "mount -a",
        f"[ -d {REPO_DIR}/.git ] || sudo -u ec2-user git clone {REPO_URL} {REPO_DIR}",
        f"sudo -u ec2-user env AWS_REGION={region} {REPO_DIR}/deploy/ec2/bootstrap.sh",
    ]


def _write_logs(group: logs.ILogGroup) -> iam.PolicyStatement:
    """New streams and lines in this one group; nothing else in CloudWatch."""
    return iam.PolicyStatement(
        actions=["logs:CreateLogStream", "logs:PutLogEvents"],
        resources=[group.log_group_arn],
    )
