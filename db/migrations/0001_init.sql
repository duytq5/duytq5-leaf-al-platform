-- Metadata for the AL loop: datasets, images, labels, rounds, dataset
-- versions, the simulation oracle's ground truth, and edge captures.
-- Applied once by `al db migrate`.
--
-- Every table has a surrogate primary key `id`. Natural keys are UNIQUE
-- constraints, so retried inserts (ON CONFLICT ... DO NOTHING) stay no-ops.
-- `id` is a bigint identity, except rounds.id: a uuid, because it is also
-- the Step Functions execution name.

-- One label list per dataset. Its order is the model's output order and is
-- copied into the model manifest.
CREATE TABLE datasets (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name        text NOT NULL UNIQUE CHECK (name ~ '^[a-z0-9][a-z0-9_-]*$'),
    labels      text[] NOT NULL CHECK (cardinality(labels) >= 2),
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE images (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id  bigint NOT NULL REFERENCES datasets (id),
    sha256      text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    s3_key      text NOT NULL,
    status      text NOT NULL CHECK (status IN ('pending', 'unlabeled', 'queued', 'labeled')),
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
    model_version    text,
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
    label       text NOT NULL,
    source      text NOT NULL CHECK (source IN ('expert', 'oracle', 'seed')),
    labeled_by  text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE NULLS NOT DISTINCT (image_id, round_id),
    CONSTRAINT seed_labels_have_no_round CHECK ((source = 'seed') = (round_id IS NULL)),
    CONSTRAINT only_experts_have_a_labeler CHECK ((source = 'expert') = (labeled_by IS NOT NULL)),
    UNIQUE (id, image_id)  -- target of dataset_version_labels' (label_id, image_id)
);

CREATE FUNCTION labels_append_only() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'labels is append-only: % is not allowed; insert a new label instead', TG_OP;
END
$$;
CREATE TRIGGER labels_append_only
    BEFORE UPDATE OR DELETE ON labels
    FOR EACH ROW EXECUTE FUNCTION labels_append_only();
CREATE TRIGGER labels_no_truncate
    BEFORE TRUNCATE ON labels
    FOR EACH STATEMENT EXECUTE FUNCTION labels_append_only();

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
    label       text NOT NULL
);

-- Edge uploads (Phase 2): the app's inference result for an image it sends.
-- capture_id is generated by the app and is the retry key. image_id is set
-- once the upload has been validated and moved to raw/.
CREATE TABLE captures (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    capture_id     uuid NOT NULL UNIQUE,
    image_id       bigint REFERENCES images (id),
    dataset_id     bigint NOT NULL REFERENCES datasets (id),
    sha256         text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    device_id      text NOT NULL,  -- from the Cognito JWT, never from the request body
    model_version  text NOT NULL,
    top1           text NOT NULL,
    confidence     real NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    probs          jsonb NOT NULL,
    captured_at    timestamptz NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now()
);
