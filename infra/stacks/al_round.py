"""The al-round state machine (design doc F4) and its Lambdas.

  MarkRunning -> Snapshot -> Train (SQS job) -> Score (SQS job) -> Select
  -> AwaitLabels -> Merge -> done
  any error -> Fail (round failed, queued images back to the pool)

Execution input: {round_id, image, train} (common.round.RoundExecution); the
start-round Lambda sets it. The execution name is the round id.

Train and score wait for the worker with a task token and no heartbeat
timeout: when the GPU machine is off, jobs wait in SQS and the round pauses.
"""

from aws_cdk import Duration, RemovalPolicy
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sqs as sqs
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as tasks
from constructs import Construct

from infra import lambda_code

STATE_MACHINE_NAME = "al-round"
# SQS keeps a job 14 days; a round waits as long for the worker.
JOB_TIMEOUT = Duration.days(14)
# How long the round waits for the expert to label its k images.
LABEL_TIMEOUT = Duration.days(30)
# Errors worth another try: Postgres or S3 briefly unreachable.
TRANSIENT_ERRORS = ["OperationalError", "EndpointConnectionError", "ConnectTimeoutError"]


class AlRound(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        vpc: ec2.IVpc,
        subnet: ec2.ISubnet,
        security_group: ec2.ISecurityGroup,
        environment: dict[str, str],
        data_bucket: s3.IBucket,
        job_queue: sqs.IQueue,
    ) -> None:
        super().__init__(scope, construct_id)
        code = lambda_code.ROUND.code()

        def step(name: str, handler: str, timeout: int = 60, memory: int = 512):
            fn = lambda_.Function(
                self,
                name,
                runtime=lambda_code.RUNTIME,
                code=code,
                handler=f"lambdas.round.{handler}",
                timeout=Duration.seconds(timeout),
                memory_size=memory,
                vpc=vpc,
                vpc_subnets=ec2.SubnetSelection(subnets=[subnet]),
                security_groups=[security_group],
                environment={**environment, "DATA_BUCKET": data_bucket.bucket_name},
                log_group=logs.LogGroup(
                    self,
                    f"{name}Logs",
                    retention=logs.RetentionDays.ONE_MONTH,
                    removal_policy=RemovalPolicy.DESTROY,
                ),
            )
            return fn

        mark_running = step("MarkRunningFn", "mark_running_handler")
        snapshot = step("SnapshotFn", "snapshot_handler", timeout=300, memory=1024)
        select = step("SelectFn", "select_handler", timeout=300, memory=1024)
        await_labels = step("AwaitLabelsFn", "await_labels_handler")
        merge = step("MergeFn", "merge_handler", timeout=120)
        fail = step("FailFn", "fail_handler")

        # S3 access by prefix (common/s3keys.py).
        data_bucket.grant_put(snapshot, "manifests/*")
        data_bucket.grant_put(snapshot, "scores/*")
        data_bucket.grant_read(select, "scores/*")
        data_bucket.grant_read_write(select, "selections/*")
        for fn in (merge, fail):
            data_bucket.grant_read(fn, "selections/*")

        def invoke(name: str, fn: lambda_.IFunction, payload: dict, result_path: str):
            task = tasks.LambdaInvoke(
                self,
                name,
                lambda_function=fn,
                payload=sfn.TaskInput.from_object(payload),
                payload_response_only=True,
                result_path=result_path,
            )
            task.add_retry(
                errors=TRANSIENT_ERRORS,
                interval=Duration.seconds(10),
                max_attempts=3,
                backoff_rate=2,
            )
            return task

        round_id = {"round_id": sfn.JsonPath.string_at("$.round_id")}
        execution = {
            **round_id,
            "image": sfn.JsonPath.string_at("$.image"),
            "train": sfn.JsonPath.object_at("$.train"),
        }

        def job(name: str, job_type: str, job_id: str, spec: dict, result_path: str):
            return tasks.SqsSendMessage(
                self,
                name,
                queue=job_queue,
                integration_pattern=sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN,
                message_body=sfn.TaskInput.from_object(
                    {
                        "job_id": sfn.JsonPath.string_at(job_id),
                        "type": job_type,
                        "image": sfn.JsonPath.string_at("$.image"),
                        "round_id": sfn.JsonPath.string_at("$.round_id"),
                        "task_token": sfn.JsonPath.task_token,
                        "spec": spec,
                    }
                ),
                task_timeout=sfn.Timeout.duration(JOB_TIMEOUT),
                result_path=result_path,
            )

        steps = [
            invoke("MarkRunning", mark_running, round_id, "$.round"),
            invoke("Snapshot", snapshot, execution, "$.snapshot"),
            job(
                "Train",
                "train",
                "$.snapshot.train.job_id",
                sfn.JsonPath.object_at("$.snapshot.train.spec"),
                "$.train_result",
            ),
            job(
                "Score",
                "score",
                "$.snapshot.score.job_id",
                {
                    "model_uri": sfn.JsonPath.string_at("$.train_result.model_uri"),
                    "pool_manifest_uri": sfn.JsonPath.string_at(
                        "$.snapshot.score.pool_manifest_uri"
                    ),
                    "outputs": sfn.JsonPath.list_at("$.snapshot.score.outputs"),
                    "output_uri": sfn.JsonPath.string_at("$.snapshot.score.output_uri"),
                },
                "$.score_result",
            ),
            invoke("Select", select, round_id, "$.select"),
            tasks.LambdaInvoke(
                self,
                "AwaitLabels",
                lambda_function=await_labels,
                integration_pattern=sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN,
                payload=sfn.TaskInput.from_object(
                    {**round_id, "task_token": sfn.JsonPath.task_token}
                ),
                task_timeout=sfn.Timeout.duration(LABEL_TIMEOUT),
                result_path=sfn.JsonPath.DISCARD,
            ),
            invoke(
                "Merge",
                merge,
                {**execution, "train_result": sfn.JsonPath.object_at("$.train_result")},
                "$.merge",
            ),
        ]

        failed = invoke("MarkFailed", fail, round_id, sfn.JsonPath.DISCARD).next(
            sfn.Fail(self, "RoundFailed", cause="see the failed step in the execution history")
        )
        chain = sfn.Chain.start(steps[0])
        for s in steps:
            s.add_catch(failed, result_path="$.error")
            if s is not steps[0]:
                chain = chain.next(s)
        chain = chain.next(sfn.Succeed(self, "RoundSucceeded"))

        self.state_machine = sfn.StateMachine(
            self,
            "StateMachine",
            state_machine_name=STATE_MACHINE_NAME,
            definition_body=sfn.DefinitionBody.from_chainable(chain),
        )
