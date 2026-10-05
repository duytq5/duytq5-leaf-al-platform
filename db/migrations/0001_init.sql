-- Metadata for the AL loop: datasets, images, labels, rounds, and the
-- simulation oracle's ground truth. Applied once by `al db migrate`.

-- One label list per dataset. Its order is the model's output order and is
-- copied into the model manifest.
CREATE TABLE datasets (
    name        text PRIMARY KEY CHECK (name ~ '^[a-z0-9][a-z0-9_-]*$'),
    labels      text[] NOT NULL CHECK (cardinality(labels) >= 2),
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE images (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset     text NOT NULL REFERENCES datasets (name),
    sha256      text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    s3_key      text NOT NULL,
    status      text NOT NULL CHECK (status IN ('pending', 'unlabeled', 'queued', 'labeled')),
    split       text NOT NULL CHECK (split IN ('train', 'val', 'test')),
    source      text NOT NULL CHECK (source IN ('seed', 'edge')),
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (dataset, sha256),
    -- The AL pool is train images with status 'unlabeled' (and 'queued' while
    -- they are being labeled). Val and test images are labeled at seeding and
    -- can never take another status, so they never enter the pool.
    CONSTRAINT eval_splits_stay_labeled CHECK (split = 'train' OR status = 'labeled')
);
CREATE INDEX images_pool_idx ON images (dataset, split, status);

CREATE TABLE rounds (
    id               uuid PRIMARY KEY,
    dataset          text NOT NULL REFERENCES datasets (name),
    mode             text NOT NULL CHECK (mode IN ('simulation', 'production')),
    -- The human-chosen selection settings from the round config YAML.
    strategy         text NOT NULL,
    strategy_cfg     jsonb NOT NULL DEFAULT '{}',
    k                integer NOT NULL CHECK (k > 0),
    seed             integer NOT NULL,
    model_version    text,
    dataset_version  text,
    status           text NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'cancelled')),
    metrics          jsonb,
    started_by       text NOT NULL,  -- Cognito username or the IAM principal of the CLI
    created_at       timestamptz NOT NULL DEFAULT now(),
    finished_at      timestamptz
);
-- One active round per dataset.
CREATE UNIQUE INDEX rounds_one_active_per_dataset
    ON rounds (dataset) WHERE status IN ('pending', 'running');

-- Labels used for training. round_id is NULL for labels that came with the
-- dataset (val and test images at seeding). An image can be relabeled in a
-- later round; (image_id, round_id) is unique so a retried merge is a no-op.
CREATE TABLE labels (
    image_id    bigint NOT NULL REFERENCES images (id),
    label       text NOT NULL,
    round_id    uuid REFERENCES rounds (id),
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE NULLS NOT DISTINCT (image_id, round_id)
);

-- Ground truth for simulation mode, read only by the oracle. Kept apart from
-- `labels` so nothing else can see a train image's label before the oracle
-- "labels" it in a round.
CREATE TABLE oracle_labels (
    image_id    bigint PRIMARY KEY REFERENCES images (id),
    label       text NOT NULL
);
