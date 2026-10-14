-- Production only (owner, 2026-10-12): rounds are labeled by experts in
-- Label Studio; there is no simulation mode any more.
ALTER TABLE rounds DROP COLUMN mode;
