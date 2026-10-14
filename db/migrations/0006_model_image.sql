-- The worker image a model was trained in (design doc F5): the release
-- exports in the same image, so it uses the same recipe code. Before this the
-- release built the tag from rounds.commit_sha, which since the ops repo split
-- (change 23) is the ops repo commit, not a platform commit with an image.
-- Set by the round's merge step and the training's record step; NULL for
-- models recorded before this migration (the release then refuses them).
ALTER TABLE model_versions ADD COLUMN image text;
