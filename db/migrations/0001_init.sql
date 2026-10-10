-- Metadata for the AL loop: datasets and their classes, images, labels, rounds, dataset
-- versions, models and their releases, the simulation oracle's ground truth,
-- and edge captures.
-- Applied once by `al db migrate`.
--
-- Every table has a surrogate primary key `id`. Natural keys are UNIQUE
-- constraints, so retried inserts (ON CONFLICT ... DO NOTHING) stay no-ops.
-- `id` is a bigint identity, except rounds.id: a uuid, because it is also
-- the Step Functions execution name.

CREATE TABLE datasets (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name        text NOT NULL UNIQUE CHECK (name ~ '^[a-z0-9][a-z0-9_-]*$'),
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- One class list per dataset. `code` is the machine name used by code, the
-- model, the manifest and API payloads; `display_name` is what users and
-- labelers read. Codes ordered by `position` (the model output index) are the
-- class list copied into the model manifest.
CREATE TABLE classes (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id    bigint NOT NULL REFERENCES datasets (id),
    code          text NOT NULL CHECK (code ~ '^[a-z0-9][a-z0-9_]*$'),
    position      integer NOT NULL CHECK (position >= 0),
    display_name  text NOT NULL,
    description   text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (dataset_id, code),
    UNIQUE (dataset_id, position)
);

CREATE TABLE images (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id  bigint NOT NULL REFERENCES datasets (id),
    sha256      text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    s3_key      text NOT NULL,
    -- 'rejected': the expert marked it "not a leaf" in Label Studio. Terminal; it
    -- gets no label and is never trained on.
    status      text NOT NULL
                CHECK (status IN ('pending', 'unlabeled', 'queued', 'labeled', 'rejected')),
    split       text NOT NULL CHECK (split IN ('train', 'val', 'test')),
    source      text NOT NULL CHECK (source IN ('seed', 'edge')),
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (dataset_id, sha256),
    -- The AL pool is train images with status 'unlabeled' (and 'queued' while
    -- they are being labeled). Val and test images are labeled at seeding and
    -- can never take another status, so they never enter the pool.
    CONSTRAINT eval_splits_stay_labeled CHECK (split = 'train' OR status = 'labeled')
);
CREATE INDEX images_pool_idx ON images (dataset_id, split, status);

-- A version is the exact labeled set a round trains on. v0 is made by the
-- seed; each round's merge step makes the next one (parent = the round's
-- base version), so rolling back is starting a round from an older version,
-- and simulated strategies that each start from v0 never see each other's
-- labels. The rows of a version are in dataset_version_labels.
CREATE TABLE dataset_versions (
    id                   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id           bigint NOT NULL REFERENCES datasets (id),
    version              integer NOT NULL CHECK (version >= 0),
    parent_version_id    bigint REFERENCES dataset_versions (id),
    created_by_round_id  uuid UNIQUE,  -- FK to rounds, added below
    manifest_uri         text,  -- manifests/{dataset}/v{n}.parquet, once exported
    created_at           timestamptz NOT NULL DEFAULT now(),
    UNIQUE (dataset_id, version),
    CONSTRAINT v0_has_no_parent CHECK (
        (version = 0) = (parent_version_id IS NULL AND created_by_round_id IS NULL)
    )
);

CREATE TABLE rounds (
    id               uuid PRIMARY KEY,
    dataset_id       bigint NOT NULL REFERENCES datasets (id),
    base_version_id  bigint NOT NULL REFERENCES dataset_versions (id),  -- what it trains on
    mode             text NOT NULL CHECK (mode IN ('simulation', 'production')),
    -- The human-chosen selection settings from the round config YAML.
    strategy         text NOT NULL,
    strategy_cfg     jsonb NOT NULL DEFAULT '{}',
    k                integer NOT NULL CHECK (k > 0),
    seed             integer NOT NULL,
    status           text NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'cancelled')),
    metrics          jsonb,
    started_by       text NOT NULL,  -- Cognito username or the IAM principal of the CLI
    -- Production mode (Phase 2): the Label Studio project for this round and
    -- the task token its webhook resumes.
    label_task_token text,
    ls_project_id    integer,
    created_at       timestamptz NOT NULL DEFAULT now(),
    finished_at      timestamptz
);
-- One active round per dataset.
CREATE UNIQUE INDEX rounds_one_active_per_dataset
    ON rounds (dataset_id) WHERE status IN ('pending', 'running');

ALTER TABLE dataset_versions
    ADD FOREIGN KEY (created_by_round_id) REFERENCES rounds (id);

-- Every label ever given, append-only: a correction is a new row in a later
-- round, never an UPDATE. Which label counts for training is decided by the
-- dataset version (dataset_version_labels). (image_id, round_id) is unique so
-- a retried merge is a no-op.
--   source 'seed'    came with the dataset (val and test at seeding); no round
--   source 'oracle'  simulation: copied from oracle_labels in a round
--   source 'expert'  production: a Label Studio user, named in labeled_by
CREATE TABLE labels (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    image_id    bigint NOT NULL REFERENCES images (id),
    round_id    uuid REFERENCES rounds (id),
    class_id    bigint NOT NULL REFERENCES classes (id),
    source      text NOT NULL CHECK (source IN ('expert', 'oracle', 'seed')),
    labeled_by  text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE NULLS NOT DISTINCT (image_id, round_id),
    CONSTRAINT seed_labels_have_no_round CHECK ((source = 'seed') = (round_id IS NULL)),
    CONSTRAINT only_experts_have_a_labeler CHECK ((source = 'expert') = (labeled_by IS NOT NULL)),
    UNIQUE (id, image_id)  -- target of dataset_version_labels' (label_id, image_id)
);

-- History tables (labels, model_releases) only ever grow.
-- Labeling progress of a round (and the merge step's "this round's labels").
CREATE INDEX labels_round_idx ON labels (round_id);

CREATE FUNCTION reject_change() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION '% is append-only: % is not allowed; insert a new row instead',
        TG_TABLE_NAME, TG_OP;
END
$$;
CREATE TRIGGER labels_append_only
    BEFORE UPDATE OR DELETE ON labels
    FOR EACH ROW EXECUTE FUNCTION reject_change();
CREATE TRIGGER labels_no_truncate
    BEFORE TRUNCATE ON labels
    FOR EACH STATEMENT EXECUTE FUNCTION reject_change();

-- The exact labeled set of each version: one label per image per version.
-- The composite FK makes sure label_id is a label of image_id.
CREATE TABLE dataset_version_labels (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    version_id  bigint NOT NULL REFERENCES dataset_versions (id),
    image_id    bigint NOT NULL REFERENCES images (id),
    label_id    bigint NOT NULL,
    UNIQUE (version_id, image_id),
    FOREIGN KEY (label_id, image_id) REFERENCES labels (id, image_id)
);

-- Ground truth for simulation mode, read only by the oracle. Kept apart from
-- `labels` so nothing else can see a train image's label before the oracle
-- "labels" it in a round.
CREATE TABLE oracle_labels (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    image_id    bigint NOT NULL UNIQUE REFERENCES images (id),
    class_id    bigint NOT NULL REFERENCES classes (id)
);

-- One model per round: a thin index over MLflow, which keeps the artifacts and
-- full logs. The worker never writes Postgres; the round's Lambda inserts the
-- row from the worker's task output, and the export job fills the ONNX columns.
-- The dataset comes through the round.
CREATE TABLE model_versions (
    id                     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    round_id               uuid NOT NULL UNIQUE REFERENCES rounds (id),
    trained_on_version_id  bigint NOT NULL REFERENCES dataset_versions (id),
    mlflow_name            text NOT NULL,
    mlflow_version         integer NOT NULL CHECK (mlflow_version >= 1),
    mlflow_run_id          text NOT NULL,
    arch                   text NOT NULL,
    train_config           jsonb NOT NULL,
    -- Class snapshot frozen when training starts and never updated:
    -- [{"code": ..., "display_name": ...}, ...] in model output order. The same
    -- JSON goes into the exported manifest.json, so the app shows the names the
    -- model was trained with even if `classes` changes later.
    labels                 jsonb NOT NULL CHECK (
        jsonb_typeof(labels) = 'array' AND jsonb_array_length(labels) >= 2
    ),
    metrics                jsonb,
    onnx_s3_key            text,
    onnx_sha256            text CHECK (onnx_sha256 ~ '^[0-9a-f]{64}$'),
    size_bytes             bigint CHECK (size_bytes > 0),
    exported_at            timestamptz,
    created_at             timestamptz NOT NULL DEFAULT now(),
    UNIQUE (mlflow_name, mlflow_version),
    CONSTRAINT export_columns_together CHECK (
        num_nulls(onnx_s3_key, onnx_sha256, size_bytes, exported_at) IN (0, 4)
    )
);

-- Every promote and rollback, append-only. The current champion of a dataset
-- is its latest release; GET /v1/devices/config reads it.
CREATE TABLE model_releases (
    id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_version_id  bigint NOT NULL REFERENCES model_versions (id),
    action            text NOT NULL CHECK (action IN ('promote', 'rollback')),
    released_by       text NOT NULL,
    released_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX model_releases_latest_idx ON model_releases (released_at DESC, id DESC);
CREATE TRIGGER model_releases_append_only
    BEFORE UPDATE OR DELETE ON model_releases
    FOR EACH ROW EXECUTE FUNCTION reject_change();
CREATE TRIGGER model_releases_no_truncate
    BEFORE TRUNCATE ON model_releases
    FOR EACH STATEMENT EXECUTE FUNCTION reject_change();

-- Edge uploads (Phase 2): the app's inference result for an image it sends.
-- capture_id is generated by the app and is the retry key. image_id is set
-- once the upload has been validated and moved to raw/.
CREATE TABLE captures (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    capture_id     uuid NOT NULL UNIQUE,
    image_id       bigint REFERENCES images (id),
    dataset_id     bigint NOT NULL REFERENCES datasets (id),
    sha256         text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    user_sub       text NOT NULL,  -- uploader's Cognito sub from the JWT, never from the body
    model_version_id bigint NOT NULL REFERENCES model_versions (id),  -- the on-device model
    -- top1 and confidence are computed by the Edge API from probs; the app
    -- does not send them.
    top1_class_id  bigint NOT NULL REFERENCES classes (id),
    confidence     real NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    probs          jsonb NOT NULL,  -- keyed by class code
    captured_at    timestamptz NOT NULL,  -- phone clock
    created_at     timestamptz NOT NULL DEFAULT now()  -- server clock; upload limits use it
);
-- Uploads per user in a time window (rate limit, blocking).
CREATE INDEX captures_user_recent_idx ON captures (user_sub, created_at);
