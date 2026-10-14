"""The model-release state machine (design doc F5) and its Lambdas.

  Prepare -> Export (SQS job, the worker) -> Record -> done
  any error -> ReleaseFailed (nothing to undo: only Record writes, in one transaction)

Execution input: common.release.ReleaseExecution, set by the ops repo's
`release` workflow when a change to configs/releases/{dataset}.yaml merges. The export
job runs in the worker image of the commit that trained the model (Prepare
finds it), so it uses the recipe code the model was trained with.

The export waits for the worker at most RELEASE_JOB_TIMEOUT, the longest a
GitHub job may wait for the execution, so the commit check shows how the
release ended. Turn the worker on before merging a release.
"""

from aws_cdk import Duration
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_sqs as sqs
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as tasks
from constructs import Construct

from infra import lambda_code
from infra.logs import kebab, lambda_logging, log_group
from infra.stacks.al_round import TRANSIENT_ERRORS

STATE_MACHINE_NAME = "model-release"
RELEASE_JOB_TIMEOUT = Duration.hours(6)

EXECUTION_FIELDS = ("dataset", "model", "version", "released_by", "commit_sha")


class ModelRelease(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        vpc: ec2.IVpc,
        subnet: ec2.ISubnet,
        security_group: ec2.ISecurityGroup,
        environment: dict[str, str],
        job_queue: sqs.IQueue,
    ) -> None:
        super().__init__(scope, construct_id)
        code = lambda_code.ROUND.code()

        def step(name: str, handler: str) -> lambda_.Function:
            return lambda_.Function(
                self,
                name,
                runtime=lambda_code.RUNTIME,
                code=code,
                handler=f"lambdas.release.{handler}",
                timeout=Duration.seconds(60),
                memory_size=256,
                vpc=vpc,
                vpc_subnets=ec2.SubnetSelection(subnets=[subnet]),
                security_groups=[security_group],
                environment=environment,
                **lambda_logging(self, f"{name}Logs", f"release/{kebab(name)}"),
            )

        execution = {f: sfn.JsonPath.string_at(f"$.{f}") for f in EXECUTION_FIELDS}
        execution["version"] = sfn.JsonPath.number_at("$.version")

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

        prepare = invoke("Prepare", step("PrepareFn", "prepare_handler"), execution, "$.prepare")
        export = tasks.SqsSendMessage(
            self,
            "Export",
            queue=job_queue,
            integration_pattern=sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN,
            message_body=sfn.TaskInput.from_object(
                {
                    "job_id": sfn.JsonPath.string_at("$.prepare.job_id"),
                    "type": "export",
                    "image": sfn.JsonPath.string_at("$.prepare.image"),
                    "task_token": sfn.JsonPath.task_token,
                    "spec": sfn.JsonPath.object_at("$.prepare.spec"),
                }
            ),
            task_timeout=sfn.Timeout.duration(RELEASE_JOB_TIMEOUT),
            result_path="$.export",
        )
        record = invoke(
            "Record",
            step("RecordFn", "record_handler"),
            {**execution, "export": sfn.JsonPath.object_at("$.export")},
            "$.record",
        )

        failed = sfn.Fail(
            self, "ReleaseFailed", cause="see the failed step in the execution history"
        )
        for s in (prepare, export, record):
            s.add_catch(failed, result_path="$.error")
        chain = prepare.next(export).next(record).next(sfn.Succeed(self, "ReleaseSucceeded"))

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
