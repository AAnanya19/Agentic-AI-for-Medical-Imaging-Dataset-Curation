"""Offline regression tests for the ISIC API downloader."""

from __future__ import annotations

import logging
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import requests
from PIL import Image

import scripts.download_isic as download_isic
from scripts.download_isic import (
    SPLIT_RATIOS,
    assign_stratified_splits,
    audit_metadata_files,
    build_candidate_inventory,
    download_image_record,
    extension_for,
    full_image_url,
    largest_remainder_allocation,
    metadata_row,
    rebuild_metadata_from_csv,
    request_with_retry,
    select_stratified_ids,
    split_counts_for,
)


def valid_image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4), color=(20, 40, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()


def response(
    status: int,
    url: str = "https://example.test/image.jpg",
    content: bytes = b"test",
) -> requests.Response:
    result = requests.Response()
    result.status_code = status
    result.url = url
    result._content = content
    result._content_consumed = True
    return result


class FakeSession:
    def __init__(self, responses: list[requests.Response]) -> None:
        self.responses = responses
        self.calls = 0
        self.urls: list[str] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, *args, **kwargs) -> requests.Response:
        self.urls.append(url)
        result = self.responses[self.calls]
        self.calls += 1
        return result


class DroppedStreamResponse:
    """A 200 response whose body connection fails after a partial chunk."""

    status_code = 200
    headers: dict[str, str] = {}

    def iter_content(self, chunk_size: int):
        yield b"partial-image-bytes"
        raise requests.exceptions.ChunkedEncodingError("connection dropped")

    def close(self) -> None:
        return None


def api_record(
    image_id: str,
    diagnosis: str,
    *,
    lesion_id: str | None = None,
    detailed_diagnosis: str | None = None,
) -> dict:
    """Build a lightweight synthetic record shaped like an API v2 listing."""
    clinical = {
        "diagnosis_3": diagnosis,
        "diagnosis_5": detailed_diagnosis or diagnosis,
    }
    if lesion_id is not None:
        clinical["lesion_id"] = lesion_id
    return {
        "isic_id": image_id,
        "files": {"full": {"url": f"https://example.test/{image_id}.jpg"}},
        "metadata": {"clinical": clinical},
    }


def synthetic_diagnosis_records() -> list[dict]:
    """Return a known 70/20/10 archive distribution for sampling tests."""
    records: list[dict] = []
    for index in range(70):
        detail = "Nevus, Dysplastic" if index % 2 else "Nevus, Clark"
        records.append(api_record(f"NEVUS_{index:03d}", "Nevus", detailed_diagnosis=detail))
    records.extend(
        api_record(f"MEL_{index:03d}", "Melanoma") for index in range(20)
    )
    records.extend(
        api_record(f"BCC_{index:03d}", "Basal cell carcinoma")
        for index in range(10)
    )
    return records


class RequestRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test.isic")

    @patch("scripts.download_isic.time.sleep")
    def test_does_not_retry_permanent_404(self, sleep) -> None:
        session = FakeSession([response(404)])
        with self.assertRaises(requests.HTTPError):
            request_with_retry(session, "https://example.test/missing", self.logger, 4, 1)
        self.assertEqual(session.calls, 1)
        sleep.assert_not_called()

    @patch("scripts.download_isic.time.sleep")
    def test_retries_transient_503(self, sleep) -> None:
        session = FakeSession([response(503), response(200)])
        result = request_with_retry(session, "https://example.test/image", self.logger, 4, 1)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(session.calls, 2)
        sleep.assert_called_once_with(1)


class ApiRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = {
            "isic_id": "ISIC_0000000",
            "files": {
                "full": {
                    "url": "https://isic-archive.s3.amazonaws.com/images/ISIC_0000000.jpg"
                }
            },
            "metadata": {
                "acquisition": {"image_type": "dermoscopic"},
                "clinical": {
                    "diagnosis_1": "Benign",
                    "diagnosis_3": "Nevus",
                    "diagnosis_5": "Nevus, Dysplastic",
                },
            },
        }

    def test_uses_api_file_url_and_extension(self) -> None:
        url = full_image_url(self.record)
        self.assertEqual(url, self.record["files"]["full"]["url"])
        self.assertEqual(extension_for(self.record, url), ".jpg")

    def test_uses_most_specific_hierarchical_diagnosis(self) -> None:
        dataset_dir = Path("/tmp/test-data/isic")
        row = metadata_row(
            self.record,
            dataset_dir / "raw/ISIC_0000000.jpg",
            dataset_dir,
            split="val",
        )
        self.assertEqual(row["diagnosis"], "Nevus, Dysplastic")
        self.assertEqual(row["split"], "val")
        self.assertIn('"image_type":"dermoscopic"', row["artifact_flags"])


class DownloadMechanicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test.isic.download")
        self.record = api_record("ISIC_1234567", "Melanoma")
        self.record["files"]["full"]["url"] = (
            "https://isic-archive.s3.amazonaws.com/images/ISIC_1234567.jpg"
        )
        self.valid_content = valid_image_bytes()

    def test_download_requests_exact_files_full_url(self) -> None:
        session = FakeSession([response(200, content=self.valid_content)])
        expected_url = self.record["files"]["full"]["url"]

        with TemporaryDirectory() as temporary_directory:
            raw_dir = Path(temporary_directory)
            image_path = download_image_record(
                session,
                self.record,
                raw_dir,
                self.logger,
                retries=0,
                timeout=1,
            )

            self.assertIsNotNone(image_path)
            self.assertEqual(session.urls, [expected_url])
            self.assertEqual(image_path.read_bytes(), self.valid_content)

    def test_downloader_source_never_constructs_download_endpoint(self) -> None:
        source = Path(download_isic.__file__).read_text(encoding="utf-8")
        self.assertNotIn("/download/", source)

    def test_missing_file_is_retried_even_when_metadata_row_exists(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory) / "isic"
            raw_dir = dataset_dir / "raw"
            raw_dir.mkdir(parents=True)
            expected_path = raw_dir / "ISIC_1234567.jpg"

            failed_session = FakeSession([response(404)])
            first_result = download_image_record(
                failed_session,
                self.record,
                raw_dir,
                self.logger,
                retries=0,
                timeout=1,
            )
            self.assertIsNone(first_result)
            self.assertFalse(expected_path.exists())

            # Simulate the stale manifest observed in the failed real run.
            pd.DataFrame(
                [{"image_id": "ISIC_1234567", "path": "raw/ISIC_1234567.jpg"}]
            ).to_csv(dataset_dir / "metadata.csv", index=False)

            retry_session = FakeSession([response(200, content=self.valid_content)])
            second_result = download_image_record(
                retry_session,
                self.record,
                raw_dir,
                self.logger,
                retries=0,
                timeout=1,
            )

            self.assertEqual(retry_session.calls, 1)
            self.assertEqual(retry_session.urls, [full_image_url(self.record)])
            self.assertEqual(second_result, expected_path)
            self.assertTrue(expected_path.is_file())

    def test_second_main_run_retries_failed_image_despite_stale_metadata(self) -> None:
        failed_session = FakeSession([response(404)])
        retry_session = FakeSession([response(200, content=self.valid_content)])

        with TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "data"
            arguments = [
                "--data-dir",
                str(data_dir),
                "--limit",
                "1",
                "--retries",
                "0",
            ]
            with (
                patch(
                    "scripts.download_isic.iter_images",
                    side_effect=lambda *args, **kwargs: iter([self.record]),
                ),
                patch(
                    "scripts.download_isic.fetch_selected_records",
                    return_value=[self.record],
                ),
                patch(
                    "scripts.download_isic.requests.Session",
                    side_effect=[failed_session, retry_session],
                ),
            ):
                download_isic.main(arguments)
                expected_path = data_dir / "isic" / "raw" / "ISIC_1234567.jpg"
                self.assertFalse(expected_path.exists())

                # Recreate the stale row that previously caused an incorrect skip.
                pd.DataFrame(
                    [
                        {
                            "image_id": "ISIC_1234567",
                            "path": "raw/ISIC_1234567.jpg",
                            "split": "train",
                            "diagnosis": "Melanoma",
                            "artifact_flags": "{}",
                        }
                    ]
                ).to_csv(data_dir / "isic" / "metadata.csv", index=False)
                download_isic.main(arguments)

            self.assertEqual(retry_session.urls, [full_image_url(self.record)])
            self.assertTrue(expected_path.is_file())
            repaired_metadata = pd.read_csv(data_dir / "isic" / "metadata.csv")
            self.assertEqual(repaired_metadata["image_id"].tolist(), ["ISIC_1234567"])

    def test_corrupt_200_response_does_not_create_file_or_metadata_row(self) -> None:
        corrupt_session = FakeSession([response(200, content=b"truncated-not-an-image")])

        with TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "data"
            arguments = ["--data-dir", str(data_dir), "--limit", "1", "--retries", "0"]
            with (
                patch(
                    "scripts.download_isic.iter_images",
                    side_effect=lambda *args, **kwargs: iter([self.record]),
                ),
                patch(
                    "scripts.download_isic.fetch_selected_records",
                    return_value=[self.record],
                ),
                patch(
                    "scripts.download_isic.requests.Session",
                    return_value=corrupt_session,
                ),
            ):
                download_isic.main(arguments)

            expected_path = data_dir / "isic" / "raw" / "ISIC_1234567.jpg"
            self.assertFalse(expected_path.exists())
            metadata = pd.read_csv(data_dir / "isic" / "metadata.csv")
            self.assertTrue(metadata.empty)

    def test_midstream_connection_drop_leaves_no_final_or_partial_file(self) -> None:
        session = FakeSession([DroppedStreamResponse()])
        with TemporaryDirectory() as temporary_directory:
            raw_dir = Path(temporary_directory)

            result = download_image_record(
                session,
                self.record,
                raw_dir,
                self.logger,
                retries=0,
                timeout=1,
            )

            self.assertIsNone(result)
            self.assertFalse((raw_dir / "ISIC_1234567.jpg").exists())
            self.assertFalse((raw_dir / "ISIC_1234567.jpg.part").exists())

    def test_main_merges_new_image_without_replacing_existing_metadata(self) -> None:
        session = FakeSession([response(200, content=self.valid_content)])
        with TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "data"
            dataset_dir = data_dir / "isic"
            raw_dir = dataset_dir / "raw"
            raw_dir.mkdir(parents=True)
            (raw_dir / "EXISTING.jpg").write_bytes(self.valid_content)
            pd.DataFrame(
                [
                    {
                        "image_id": "EXISTING",
                        "path": "raw/EXISTING.jpg",
                        "split": "train",
                        "diagnosis": "Nevus",
                        "artifact_flags": "{}",
                    }
                ]
            ).to_csv(dataset_dir / "metadata.csv", index=False)

            with (
                patch(
                    "scripts.download_isic.iter_images",
                    side_effect=lambda *args, **kwargs: iter([self.record]),
                ),
                patch(
                    "scripts.download_isic.fetch_selected_records",
                    return_value=[self.record],
                ),
                patch("scripts.download_isic.requests.Session", return_value=session),
            ):
                download_isic.main(
                    ["--data-dir", str(data_dir), "--limit", "1", "--retries", "0"]
                )

            merged = pd.read_csv(dataset_dir / "metadata.csv")
            self.assertEqual(merged["image_id"].tolist(), ["EXISTING", "ISIC_1234567"])


class RepairModeTests(unittest.TestCase):
    def test_repair_identifies_only_orphaned_metadata_rows(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "data"
            dataset_dir = data_dir / "isic"
            raw_dir = dataset_dir / "raw"
            raw_dir.mkdir(parents=True)
            (raw_dir / "valid.jpg").write_bytes(valid_image_bytes())
            metadata_path = dataset_dir / "metadata.csv"
            pd.DataFrame(
                [
                    {"image_id": "VALID", "path": "raw/valid.jpg"},
                    {"image_id": "ORPHAN", "path": "raw/missing.jpg"},
                ]
            ).to_csv(metadata_path, index=False)

            total_rows, orphaned = audit_metadata_files(dataset_dir, metadata_path)

            self.assertEqual(total_rows, 2)
            self.assertEqual([row["image_id"] for row in orphaned], ["ORPHAN"])
            with patch("scripts.download_isic.requests.Session") as session_class:
                download_isic.main(["--data-dir", str(data_dir), "--repair"])
            session_class.assert_not_called()
            self.assertEqual(len(pd.read_csv(metadata_path)), 2)

    def test_rebuild_merges_only_valid_cached_rows_and_creates_backup(self) -> None:
        image_bytes = valid_image_bytes()
        with TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "data"
            dataset_dir = data_dir / "isic"
            raw_dir = dataset_dir / "raw"
            raw_dir.mkdir(parents=True)
            (raw_dir / "current.jpg").write_bytes(image_bytes)
            (raw_dir / "recovered.jpg").write_bytes(image_bytes)
            metadata_path = dataset_dir / "metadata.csv"
            columns = ["image_id", "path", "split", "diagnosis", "artifact_flags"]
            pd.DataFrame(
                [["CURRENT", "raw/current.jpg", "train", "Nevus", "{}"]],
                columns=columns,
            ).to_csv(metadata_path, index=False)
            recovery_path = dataset_dir / "cached.csv"
            pd.DataFrame(
                [
                    ["RECOVERED", "raw/recovered.jpg", "unassigned", "Nevus", "{}"],
                    ["MISSING", "raw/missing.jpg", "unassigned", "Nevus", "{}"],
                ],
                columns=columns,
            ).to_csv(recovery_path, index=False)

            total, added = rebuild_metadata_from_csv(
                data_dir,
                str(recovery_path),
                logging.getLogger("test.isic.rebuild"),
            )

            self.assertEqual((total, added), (2, 1))
            rebuilt = pd.read_csv(metadata_path)
            self.assertEqual(rebuilt["image_id"].tolist(), ["CURRENT", "RECOVERED"])
            self.assertEqual(
                len(list(dataset_dir.glob("metadata.before_rebuild_*.csv"))),
                1,
            )


class StratifiedSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inventory = build_candidate_inventory(
            synthetic_diagnosis_records(),
            stratify_by="diagnosis",
        )

    def test_proportional_sampling_matches_known_archive_distribution(self) -> None:
        selected, allocation = select_stratified_ids(
            self.inventory,
            limit=20,
            balanced=False,
            seed=42,
        )

        self.assertEqual(
            allocation,
            {"Basal cell carcinoma": 2, "Melanoma": 4, "Nevus": 14},
        )
        self.assertEqual({key: len(value) for key, value in selected.items()}, allocation)
        self.assertEqual(sum(allocation.values()), 20)

    def test_balanced_sampling_allocates_equally_with_exact_limit(self) -> None:
        selected, allocation = select_stratified_ids(
            self.inventory,
            limit=20,
            balanced=True,
            seed=42,
        )

        # Lexical tie-breaking assigns the two remainder slots to BCC and melanoma.
        self.assertEqual(
            allocation,
            {"Basal cell carcinoma": 7, "Melanoma": 7, "Nevus": 6},
        )
        self.assertEqual({key: len(value) for key, value in selected.items()}, allocation)
        self.assertEqual(sum(allocation.values()), 20)

    def test_same_seed_is_reproducible_independent_of_inventory_order(self) -> None:
        selected_a, allocation_a = select_stratified_ids(
            self.inventory,
            limit=30,
            balanced=False,
            seed=2026,
        )
        reversed_inventory = {
            category: list(reversed(image_ids))
            for category, image_ids in reversed(list(self.inventory.items()))
        }
        selected_b, allocation_b = select_stratified_ids(
            reversed_inventory,
            limit=30,
            balanced=False,
            seed=2026,
        )

        self.assertEqual(allocation_a, allocation_b)
        self.assertEqual(selected_a, selected_b)


class StratifiedSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records: list[dict] = []
        categories = (
            ("Nevus", "NEV"),
            ("Melanoma", "MEL"),
            ("Basal cell carcinoma", "BCC"),
        )
        # Twenty images per category, arranged as ten two-image lesion groups.
        for diagnosis, prefix in categories:
            for lesion_index in range(10):
                lesion_id = f"{prefix}_LESION_{lesion_index:02d}"
                for image_index in range(2):
                    self.records.append(
                        api_record(
                            f"{prefix}_{lesion_index:02d}_{image_index}",
                            diagnosis,
                            lesion_id=lesion_id,
                        )
                    )

    def test_split_is_diagnosis_stratified_and_lesion_grouped(self) -> None:
        assignment = assign_stratified_splits(
            self.records,
            stratify_by="diagnosis",
            seed=73,
        )

        lesion_splits: dict[str, set[str]] = {}
        for record in self.records:
            lesion_id = record["metadata"]["clinical"]["lesion_id"]
            lesion_splits.setdefault(lesion_id, set()).add(assignment[record["isic_id"]])
        self.assertTrue(lesion_splits)
        self.assertTrue(all(len(splits) == 1 for splits in lesion_splits.values()))

        counts = split_counts_for(self.records, assignment, stratify_by="diagnosis")
        target = largest_remainder_allocation(SPLIT_RATIOS, 20)
        for category_counts in counts.values():
            for split, expected in target.items():
                # Whole two-image lesions make exact odd val/test targets impossible.
                self.assertLessEqual(abs(category_counts[split] - expected), 2)

    def test_same_seed_reproduces_split_assignment(self) -> None:
        assignment_a = assign_stratified_splits(
            self.records,
            stratify_by="diagnosis",
            seed=90210,
        )
        assignment_b = assign_stratified_splits(
            list(reversed(self.records)),
            stratify_by="diagnosis",
            seed=90210,
        )

        self.assertEqual(assignment_a, assignment_b)

    def test_missing_lesion_ids_are_split_as_independent_images(self) -> None:
        records = [api_record(f"NO_LESION_{index:02d}", "Nevus") for index in range(20)]

        assignment = assign_stratified_splits(
            records,
            stratify_by="diagnosis",
            seed=11,
        )

        counts = split_counts_for(records, assignment, stratify_by="diagnosis")["Nevus"]
        self.assertEqual(counts, {"train": 14, "val": 3, "test": 3})


if __name__ == "__main__":
    unittest.main()
