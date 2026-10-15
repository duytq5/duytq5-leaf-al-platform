"""The al-round state machine (design doc F4) and its Lambdas.

  MarkRunning -> Snapshot -> Train (SQS job) -> Score (SQS job) -> Select
  -> AwaitLabels -> Merge -> CloseLabelProject -> done
  any error -> Fail (round failed, queued images back to the pool)
  CloseLabelProject error -> LabelProjectNotClosed (the round stays succeeded)

A selection-only round (execution input `train` is null, design doc change 24)
skips Train and scores with the round's own `model`; a random round with no
model (`model` null too) skips Score as well and selects from the pool list.

AwaitLabels (lambdas/labeling.py) puts the selected images into the round's
Label Studio project and waits; RecordLabelFn stores each submitted label for
the webhook Lambda outside the VPC (infra/stacks/platform.py), which resumes
the round once all k images are done. Once merged, CloseLabelProject exports
the project to labelstudio/<round id>.json in the data bucket (kept, no
expiry) and deletes it from Label Studio.

Execution input: {round_id, image, train, model} (common.round.RoundExecution); the
start-round Lambda sets it. The execution name is the round id.

Train and score wait for the worker with a task token and no heartbeat
timeout: when the GPU machine is off, jobs wait in SQS and the round pauses.
"""

from aws_cdk import Duration
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sqs as sqs
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as tasks
from constructs import Construct

from infra import lambda_code
from infra.logs import kebab, lambda_logging, log_group

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
        label_studio_environment: dict[str, str],
        data_bucket: s3.IBucket,
        job_queue: sqs.IQueue,
    ) -> None:
        super().__init__(scope, construct_id)
        code = lambda_code.ROUND.code()

        def step(
            name: str,
            handler: str,
            timeout: int = 60,
            memory: int = 512,
            module: str = "round",
            extra_environment: dict[str, str] | None = None,
        ):
            fn = lambda_.Function(
                self,
                name,
                runtime=lambda_code.RUNTIME,
                code=code,
                handler=f"lambdas.{module}.{handler}",
                timeout=Duration.seconds(timeout),
                memory_size=memory,
                vpc=vpc,
                vpc_subnets=ec2.SubnetSelection(subnets=[subnet]),
                security_groups=[security_group],
                environment={
                    **environment,
                    **(extra_environment or {}),
                    "DATA_BUCKET": data_bucket.bucket_name,
                },
                **lambda_logging(self, f"{name}Logs", f"round/{kebab(name)}"),
            )
            return fn

        mark_running = step("MarkRunningFn", "mark_running_handler")
        snapshot = step("SnapshotFn", "snapshot_handler", timeout=300, memory=1024)
        select = step("SelectFn", "select_handler", timeout=300, memory=1024)
        # The webhook URL is added by the stack once the HTTP API exists.
        self.start_labeling_fn = await_labels = step(
            "AwaitLabelsFn",
            "start_handler",
            timeout=120,  # a handful of Label Studio calls, 20 s each at most
            module="labeling",
            extra_environment=label_studio_environment,
        )
        self.record_label_fn = record_label = step(
            "RecordLabelFn",
            "record_handler",
            module="labeling",
            extra_environment=label_studio_environment,
        )
        merge = step("MergeFn", "merge_handler", timeout=120)
        close_project = step(
            "CloseLabelProjectFn",
            "close_handler",
            timeout=120,  # export, put, read back, delete
            memory=1024,
            module="labeling",
            extra_environment=label_studio_environment,
        )
        fail = step("FailFn", "fail_handler")

        # S3 access by prefix (common/s3keys.py).
        data_bucket.grant_put(snapshot, "manifests/*")
        data_bucket.grant_put(snapshot, "scores/*")
        data_bucket.grant_read(select, "scores/*")
        data_bucket.grant_read_write(select, "selections/*")
        for fn in (merge, fail, record_label):
            data_bucket.grant_read(fn, "selections/*")
        for prefix in ("selections/*", "scores/*"):
            data_bucket.grant_read(await_labels, prefix)
        data_bucket.grant_put(close_project, "labelstudio/*")
        data_bucket.grant_read(close_project, "labelstudio/*")

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
            "model": sfn.JsonPath.string_at("$.model"),
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

        mark = invoke("MarkRunning", mark_running, round_id, "$.round")
        snap = invoke("Snapshot", snapshot, execution, "$.snapshot")
        train = job(
            "Train",
            "train",
            "$.snapshot.train.job_id",
            sfn.JsonPath.object_at("$.snapshot.train.spec"),
            "$.train_result",
        )
        score = job(
            "Score",
            "score",
            "$.snapshot.score.job_id",
            {
                "model_uri": sfn.JsonPath.string_at("$.scoring.model_uri"),
                "pool_manifest_uri": sfn.JsonPath.string_at("$.snapshot.score.pool_manifest_uri"),
                "outputs": sfn.JsonPath.list_at("$.snapshot.score.outputs"),
                "output_uri": sfn.JsonPath.string_at("$.snapshot.score.output_uri"),
            },
            "$.score_result",
        )
        select_ = invoke("Select", select, round_id, "$.select")
        labels = self._await_labels(await_labels, round_id)
        # No payload: the Lambda gets the whole state, since train_result
        # exists only when Train ran.
        merge_ = tasks.LambdaInvoke(
            self,
            "Merge",
            lambda_function=merge,
            payload_response_only=True,
            result_path="$.merge",
        )
        merge_.add_retry(
            errors=TRANSIENT_ERRORS, interval=Duration.seconds(10), max_attempts=3, backoff_rate=2
        )

        close = invoke("CloseLabelProject", close_project, round_id, "$.close")
        close.add_retry(
            errors=["LabelStudioError"],
            interval=Duration.seconds(30),
            max_attempts=4,
            backoff_rate=2,
        )
        # The round succeeded already, so no MarkFailed; the execution still
        # fails so the leftover project shows.
        close.add_catch(
            sfn.Fail(
                self,
                "LabelProjectNotClosed",
                cause="round succeeded; its Label Studio project was not archived and deleted",
            ),
            result_path="$.error",
        )

        def scoring_model(name: str, path: str) -> sfn.Pass:
            return sfn.Pass(
                self,
                name,
                parameters={"model_uri": sfn.JsonPath.string_at(path)},
                result_path="$.scoring",
            )

        failed = invoke("MarkFailed", fail, round_id, sfn.JsonPath.DISCARD).next(
            sfn.Fail(self, "RoundFailed", cause="see the failed step in the execution history")
        )
        for s in (mark, snap, train, score, select_, labels, merge_):
            s.add_catch(failed, result_path="$.error")

        score.next(select_).next(labels).next(merge_).next(close).next(
            sfn.Succeed(self, "RoundSucceeded")
        )
        chain = (
            sfn.Chain.start(mark)
            .next(snap)
            .next(
                sfn.Choice(self, "TrainOrNot")
                .when(
                    sfn.Condition.is_null("$.train"),
                    # A random round with no model needs no scores: no worker.
                    sfn.Choice(self, "ScoreOrNot")
                    .when(sfn.Condition.is_null("$.model"), select_)
                    .otherwise(
                        scoring_model("UseRoundModel", "$.snapshot.score.model_uri").next(score)
                    ),
                )
                .otherwise(
                    train.next(scoring_model("UseTrainedModel", "$.train_result.model_uri")).next(
                        score
                    )
                )
            )
        )

        self.state_machine = sfn.StateMachine(
            self,
            "StateMachine",
            state_machine_name=STATE_MACHINE_NAME,
            definition_body=sfn.DefinitionBody.from_chainable(chain),
            # Every event with its input and output: a failed step's error and
            # the round_id it ran for (`al logs <round_id>`).
            logs=sfn.LogOptions(
                destination=log_group(self, "StateMachineLogs", f"states/{STATE_MACHINE_NAME}"),
                level=sfn.LogLevel.ALL,
                include_execution_data=True,
            ),
        )

    def _await_labels(self, fn: lambda_.IFunction, round_id: dict) -> tasks.LambdaInvoke:
        task = tasks.LambdaInvoke(
            self,
            "AwaitLabels",
            lambda_function=fn,
            integration_pattern=sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN,
            payload=sfn.TaskInput.from_object({**round_id, "task_token": sfn.JsonPath.task_token}),
            task_timeout=sfn.Timeout.duration(LABEL_TIMEOUT),
            result_path=sfn.JsonPath.DISCARD,
        )
        # Label Studio may be restarting, or the host just started. A retry
        # gets a new token; the step stores it and reuses the project.
        task.add_retry(
            errors=["LabelStudioError", *TRANSIENT_ERRORS],
            interval=Duration.seconds(30),
            max_attempts=4,
            backoff_rate=2,
        )
        return task
