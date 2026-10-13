-- What the app declared about the image of a capture (POST /v1/captures).
-- The presigned PUT is signed for this type and size, and the upload
-- validation Lambda checks the file against them. A retry with the same
-- capture_id must declare the same values, or it is a conflict (409).
-- Nothing has written captures yet, so the columns can be NOT NULL at once.
ALTER TABLE captures
    ADD COLUMN content_type text NOT NULL CHECK (content_type IN ('image/jpeg', 'image/png')),
    ADD COLUMN size_bytes   bigint NOT NULL CHECK (size_bytes > 0);
