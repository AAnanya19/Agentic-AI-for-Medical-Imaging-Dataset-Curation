"""Regression tests for shared downloader metadata merge semantics."""

from __future__ import annotations

import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.common import (
    load_existing_metadata,
    merge_metadata_frames,
    write_metadata_atomic,
)


class MetadataMergeTests(unittest.TestCase):
    columns = ("image_id", "path", "split")

    def test_existing_rows_win_and_unseen_rows_append(self) -> None:
        existing = pd.DataFrame(
            [
                {"image_id": "A", "path": "raw/a.jpg", "split": "train"},
                {"image_id": "B", "path": "raw/b-old.jpg", "split": "val"},
            ]
        )
        new = pd.DataFrame(
            [
                {"image_id": "B", "path": "raw/b-new.jpg", "split": "test"},
                {"image_id": "C", "path": "raw/c.jpg", "split": "train"},
            ]
        )

        merged = merge_metadata_frames(
            existing,
            new,
            self.columns,
            ("image_id",),
        )

        self.assertEqual(merged["image_id"].tolist(), ["A", "B", "C"])
        self.assertEqual(merged.loc[1, "path"], "raw/b-old.jpg")
        self.assertEqual(merged.loc[1, "split"], "val")

    def test_explicit_reprocessed_key_updates_in_place(self) -> None:
        existing = pd.DataFrame(
            [{"image_id": "A", "path": "raw/old.jpg", "split": "train"}]
        )
        new = pd.DataFrame(
            [{"image_id": "A", "path": "raw/new.jpg", "split": "val"}]
        )

        merged = merge_metadata_frames(
            existing,
            new,
            self.columns,
            ("image_id",),
            update_keys={("A",)},
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.iloc[0].to_dict(), new.iloc[0].to_dict())

    def test_fresh_ignores_existing_manifest_and_atomic_writer_replaces_it(self) -> None:
        logger = logging.getLogger("test.metadata.merge")
        with TemporaryDirectory() as temporary_directory:
            metadata_path = Path(temporary_directory) / "metadata.csv"
            pd.DataFrame(
                [{"image_id": "OLD", "path": "raw/old.jpg", "split": "train"}]
            ).to_csv(metadata_path, index=False)

            with self.assertLogs(logger, level="WARNING") as captured:
                fresh = load_existing_metadata(
                    metadata_path,
                    self.columns,
                    ("image_id",),
                    fresh=True,
                    logger=logger,
                )
            self.assertTrue(fresh.empty)
            self.assertIn("FRESH MODE", "\n".join(captured.output))

            replacement = pd.DataFrame(
                [{"image_id": "NEW", "path": "raw/new.jpg", "split": "test"}]
            )
            write_metadata_atomic(replacement, metadata_path)
            self.assertEqual(pd.read_csv(metadata_path)["image_id"].tolist(), ["NEW"])


if __name__ == "__main__":
    unittest.main()
