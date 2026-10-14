-- Standalone trainings (design doc change 24): the ops repo's start-training
-- workflow trains a model on a dataset version without a round. id is
-- uuid5(commit_sha + config path), so a re-run workflow cannot start it twice.
-- A training changes no data, so it does not block rounds.
CREATE TABLE trainings (
    id            uuid PRIMARY KEY,
    dataset_id    bigint NOT NULL REFERENCES datasets (id),
    version_id    bigint NOT NULL REFERENCES dataset_versions (id),  -- what it trains on
    train_config  jsonb NOT NULL,  -- {recipe, params, base_model}, params validated
    status        text NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'running', 'succeeded', 'failed')),
    started_by    text NOT NULL,  -- GitHub login that merged the training file
    commit_sha    text NOT NULL CHECK (commit_sha ~ '^[0-9a-f]{40}$'),
    created_at    timestamptz NOT NULL DEFAULT now(),
    finished_at   timestamptz
);

-- A model now comes from a round or a training, so it names its dataset
-- itself (filled from the round for the models trained so far).
ALTER TABLE model_versions ADD COLUMN dataset_id bigint REFERENCES datasets (id);
UPDATE model_versions mv SET dataset_id = r.dataset_id FROM rounds r WHERE r.id = mv.round_id;
ALTER TABLE model_versions ALTER COLUMN dataset_id SET NOT NULL;
ALTER TABLE model_versions ALTER COLUMN round_id DROP NOT NULL;
ALTER TABLE model_versions ADD COLUMN training_id uuid UNIQUE REFERENCES trainings (id);
ALTER TABLE model_versions ADD CONSTRAINT model_versions_one_source
    CHECK (num_nonnulls(round_id, training_id) = 1);
CREATE INDEX model_versions_dataset_idx ON model_versions (dataset_id, created_at DESC);
