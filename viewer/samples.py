"""FiftyOne samples and datasets from image records. Only this module imports
FiftyOne, which is installed in the viewer image only."""

from pathlib import Path

import fiftyone as fo

from viewer.records import ImageRecord


def to_sample(record: ImageRecord, path: Path) -> fo.Sample:
    """One sample per image. Tags (split, status, source) give one-click filters
    in the app; the other fields filter in the sidebar."""

    def guess(code: str | None, confidence: float | None = None) -> fo.Classification | None:
        return fo.Classification(label=code, confidence=confidence) if code else None

    return fo.Sample(
        filepath=str(path),
        tags=[record.split, record.status, record.source],
        image_id=record.image_id,
        s3_key=record.s3_key,
        split=record.split.value,
        status=record.status.value,
        source=record.source.value,
        created_at=record.created_at,
        ground_truth=guess(record.label),
        label_source=record.label_source,
        labeled_by=record.labeled_by,
        label_round=str(record.label_round) if record.label_round else None,
        versions=record.versions,
        first_version=record.versions[0] if record.versions else None,
        new_label=guess(record.new_label),
        new_label_round=str(record.new_label_round) if record.new_label_round else None,
        app_prediction=guess(record.app_prediction, record.app_confidence),
    )


def sync_dataset(name: str, samples: list[fo.Sample]) -> fo.Dataset:
    """Make the FiftyOne dataset `name` hold exactly these samples, matched by
    image_id: new images are added, changed ones updated, gone ones removed.
    The dataset stays in place, so an open browser tab keeps working."""
    dataset = fo.load_dataset(name) if fo.dataset_exists(name) else fo.Dataset(name)
    dataset.persistent = True
    if samples:
        # Lists (tags, versions) are replaced, not merged: an image changes status.
        dataset.merge_samples(samples, key_field="image_id", merge_lists=False)
    keep = {s["image_id"] for s in samples}
    gone = [s.id for s in dataset.select_fields("image_id") if s["image_id"] not in keep]
    if gone:
        dataset.delete_samples(gone)
    return dataset
