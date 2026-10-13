"""The single CDK stack for the POC.

This stack holds the core infrastructure: S3 buckets (al-data, al-models), the
SQS job queue + DLQ, the worker's IAM user, the EC2 host with its retained data
volume, SSM parameters for non-secret settings, the Budget alert, and the
Cognito user pool that signs people in, and the al-round state machine with
its Lambdas (infra/stacks/al_round.py). Later PRs add the API here.

The EC2 host has no inbound ports. The owner and the worker reach MLflow (5000)
and Postgres (5432) through SSM Session Manager port forwarding, so access
depends on IAM, not on an IP address.

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
"""

from pathlib import Path

from aws_cdk import (
    Acknowledgment,
    Annotations,
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
from aws_cdk import aws_budgets as budgets
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sqs as sqs
from aws_cdk import aws_ssm as ssm
from constructs import Construct

from infra.stacks.al_round import AlRound

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

AL2023_AMI_PARAM = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
DOCKER_COMPOSE_VERSION = "v2.29.7"
DOCKER_BUILDX_VERSION = "v0.17.1"
REPO_URL = "https://github.com/duytq5/duytq5-leaf-al-platform.git"
REPO_DIR = "/home/ec2-user/duytq5-leaf-al-platform"
# The host's subnet already spans the VPC's 10.0.0.0/16, so the Lambdas' private
# subnet sits on a second range; the host and its subnet stay as they are.
LAMBDA_CIDR = "10.1.0.0/24"

LAMBDAS_DIR = Path(__file__).resolve().parents[2] / "lambdas"

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
        self._budget(ctx("budgetEmail"), float(ctx("budgetUsd") or 20))
        self._auth(ctx)

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
        policy.add_statements(
            iam.PolicyStatement(
                sid="ListMlflowArtifacts",
                actions=["s3:ListBucket"],
                resources=[models.bucket_arn],
                conditions={"StringLike": {"s3:prefix": ["mlflow-artifacts/*"]}},
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

    def _al_round(self) -> None:
        al_round = AlRound(
            self,
            "AlRound",
            vpc=self.vpc,
            subnet=self.lambda_subnet,
            security_group=self.lambda_sg,
            environment={
                "PG_HOSTADDR": self.host.instance_private_ip,
                "PG_PASSWORD": SecretValue.secrets_manager(
                    POSTGRES_PASSWORD_SECRET
                ).unsafe_unwrap(),
                "PG_CA": ssm.StringParameter.value_for_string_parameter(self, SSM_POSTGRES_TLS_CA),
            },
            data_bucket=self.data_bucket,
            job_queue=self.job_queue,
        )
        ssm.StringParameter(
            self,
            "AlRoundArnParam",
            parameter_name=SSM_AL_ROUND_ARN,
            string_value=al_round.state_machine.state_machine_arn,
        )
        CfnOutput(self, "AlRoundStateMachineArn", value=al_round.state_machine.state_machine_arn)

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
    ) -> cognito.UserPoolClient:
        providers = [cognito.UserPoolClientIdentityProvider.COGNITO]
        if google:
            providers.append(cognito.UserPoolClientIdentityProvider.GOOGLE)
        client = self.user_pool.add_client(
            construct_id,
            user_pool_client_name=name,
            generate_secret=False,
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

    def _post_confirmation_fn(self) -> lambda_.Function:
        log_group = logs.LogGroup(
            self,
            "PostConfirmationLogs",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )
        fn = lambda_.Function(
            self,
            "PostConfirmationFn",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=lambda_.Code.from_asset(
                str(LAMBDAS_DIR / "post_confirmation"), exclude=["__pycache__", "*.pyc"]
            ),
            timeout=Duration.seconds(10),
            log_group=log_group,
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
