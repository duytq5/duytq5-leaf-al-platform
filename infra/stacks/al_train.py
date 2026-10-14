"""The al-train state machine (design doc change 24) and its Lambdas: a
standalone training, started by the ops repo's start-training workflow.

  MarkRunning -> Snapshot -> Train (SQS job) -> Record -> done
  any error -> Fail (training failed)

Execution input: {training_id, image} (common.training.TrainingExecution); the
start-training Lambda sets it. The execution name is the training id. The
train job is the same as a round's (same worker code and recipe), with
training_id instead of round_id, and waits for the worker without a heartbeat
timeout: when the GPU machine is off, the job waits in SQS.
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
from infra.stacks.al_round import JOB_TIMEOUT, TRANSIENT_ERRORS

STATE_MACHINE_NAME = "al-train"


class AlTrain(Construct):
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
            return lambda_.Function(
                self,
                name,
                runtime=lambda_code.RUNTIME,
                code=code,
                handler=f"lambdas.training.{handler}",
                timeout=Duration.seconds(timeout),
                memory_size=memory,
                vpc=vpc,
                vpc_subnets=ec2.SubnetSelection(subnets=[subnet]),
                security_groups=[security_group],
                environment={**environment, "DATA_BUCKET": data_bucket.bucket_name},
                **lambda_logging(self, f"{name}Logs", f"training/{kebab(name)}"),
            )

        mark_running = step("MarkRunningFn", "mark_running_handler")
        snapshot = step("SnapshotFn", "snapshot_handler", timeout=300, memory=1024)
        record = step("RecordFn", "record_handler")
        fail = step("FailFn", "fail_handler")
        data_bucket.grant_put(snapshot, "manifests/*")

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

        training_id = {"training_id": sfn.JsonPath.string_at("$.training_id")}
        train = tasks.SqsSendMessage(
            self,
            "Train",
            queue=job_queue,
            integration_pattern=sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN,
            message_body=sfn.TaskInput.from_object(
                {
                    "job_id": sfn.JsonPath.string_at("$.snapshot.job_id"),
                    "type": "train",
                    "image": sfn.JsonPath.string_at("$.image"),
                    "training_id": sfn.JsonPath.string_at("$.training_id"),
                    "task_token": sfn.JsonPath.task_token,
                    "spec": sfn.JsonPath.object_at("$.snapshot.spec"),
                }
            ),
            task_timeout=sfn.Timeout.duration(JOB_TIMEOUT),
            result_path="$.train_result",
        )
        steps = [
            invoke("MarkRunning", mark_running, training_id, "$.training"),
            invoke("Snapshot", snapshot, training_id, "$.snapshot"),
            train,
            invoke(
                "Record",
                record,
                {
                    **training_id,
                    "image": sfn.JsonPath.string_at("$.image"),
                    "train_result": sfn.JsonPath.object_at("$.train_result"),
                },
                "$.record",
            ),
        ]
        failed = invoke("MarkFailed", fail, training_id, sfn.JsonPath.DISCARD).next(
            sfn.Fail(self, "TrainingFailed", cause="see the failed step in the execution history")
        )
        chain = sfn.Chain.start(steps[0])
        for s in steps:
            s.add_catch(failed, result_path="$.error")
            if s is not steps[0]:
                chain = chain.next(s)
        chain = chain.next(sfn.Succeed(self, "TrainingSucceeded"))

        self.state_machine = sfn.StateMachine(
            self,
            "StateMachine",
            state_machine_name=STATE_MACHINE_NAME,
            definition_body=sfn.DefinitionBody.from_chainable(chain),
            logs=sfn.LogOptions(
                destination=log_group(self, "StateMachineLogs", f"states/{STATE_MACHINE_NAME}"),
                level=sfn.LogLevel.ALL,
                include_execution_data=True,
            ),
        )
