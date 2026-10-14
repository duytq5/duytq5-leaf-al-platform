-- Selection-only rounds (design doc change 24): a round without a train
-- section scores the pool with an existing model. score_model is that model
-- (models:/name/N or models:/name@alias); NULL for a round that trains, whose
-- model is its model_versions row.
ALTER TABLE rounds ADD COLUMN score_model text;
