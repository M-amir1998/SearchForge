"""Ingestion formats/safety, normalization and chunking."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from searchforge.chunking.chunkers import (
    FixedTokenApproxChunker,
    SectionChunker,
    SentenceChunker,
    Span,
    build_chunker,
    merge_short_spans,
)
from searchforge.config.loader import validate_config
from searchforge.config.models import ChunkingConfig, ChunkStrategy, IngestionLimits
from searchforge.domain.document import Document, DocumentMetadata, DocumentSource, SourceFormat
from searchforge.errors import CorpusNotFound, IngestionError, UnsafePathError
from searchforge.ingestion.readers import iter_corpus
from searchforge.ingestion.safety import IngestionReport, decode_bytes, iter_source_files
from searchforge.parsing.formats import html_to_text, markdown_headings, parse_markdown
from searchforge.parsing.normalize import normalize_query_text, normalize_text

pytestmark = pytest.mark.unit


def _loaded(tmp_path: Path, sources: list[dict[str, object]], **limits: object):  # type: ignore[no-untyped-def]
    return validate_config(
        {
            "workspace": str(tmp_path / "ws"),
            "corpus": {"id": "c", "sources": sources, "limits": limits},
            "index": {"name": "i"},
        },
        tmp_path,
    )


def test_normalization_is_conservative() -> None:
    raw = "Title\r\n\r\n\r\n\r\n\r\nBody text  with\ttabs\x00 and RetryPolicy.max_attempts \u200b\n"
    out = normalize_text(raw)
    assert out == "Title\n\n\nBody text with tabs and RetryPolicy.max_attempts"
    assert normalize_text("ﬁle") == "ﬁle"  # NFC keeps ligature (identifiers preserved)
    assert normalize_query_text("  hello \n world ") == "hello world"


def test_markdown_front_matter_title_and_headings() -> None:
    md = parse_markdown("---\ntitle: T\ncategory: ops\n---\n# Heading\n\ntext\n")
    assert md.title == "T" and md.front_matter["category"] == "ops" and md.body.startswith("# Heading")
    assert parse_markdown("# First\n\n## Second").title == "First"
    heads = markdown_headings("# A\n```\n# not a heading\n```\n## B\n")
    assert [(lvl, t) for _, lvl, t in heads] == [(1, "A"), (2, "B")]


def test_html_extraction_drops_scripts() -> None:
    text, title = html_to_text(
        "<html><head><title>Doc</title><script>alert(1)</script></head>"
        "<body><h1>Head</h1><p>Visible &amp; safe</p><style>p{}</style></body></html>"
    )
    assert title == "Doc"
    assert "alert" not in text and "p{}" not in text
    assert "# Head" in text and "Visible & safe" in text


def test_all_formats_stream(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    (root / "a.txt").write_text("Plain title\n\nplain body", encoding="utf-8")
    (root / "b.md").write_bytes(b"---\r\nid: MD-1\r\ncategory: ops\r\n---\r\n# MD\r\n\r\nmarkdown body\r\n")
    (root / "c.json").write_text(
        json.dumps([{"id": "J1", "text": "json one"}, {"id": "J2", "text": "json two"}]), encoding="utf-8"
    )
    (root / "d.jsonl").write_text(
        '{"id": "L1", "text": "line one", "year": 2024}\nnot json\n{"id": "L2", "text": ""}\n', encoding="utf-8"
    )
    (root / "e.csv").write_text("id,title,text\nC1,CSV title,csv body\n", encoding="utf-8")
    pq.write_table(pa.Table.from_pylist([{"id": "P1", "text": "parquet body", "n": 3}]), root / "f.parquet")
    (root / "g.html").write_text("<title>H</title><p>html body</p>", encoding="utf-8")
    (root / "h.bin").write_bytes(b"\x00\x01")
    report = IngestionReport()
    docs = list(iter_corpus(_loaded(tmp_path, [{"id": "s", "path": str(root)}]), report))
    ids = {d.document_id for d in docs}
    assert {"MD-1", "J1", "J2", "L1", "C1", "P1"} <= ids
    md = next(d for d in docs if d.document_id == "MD-1")
    assert md.metadata.attributes["category"] == "ops" and md.metadata.title == "MD"
    assert next(d for d in docs if d.document_id == "L1").metadata.attributes["year"] == 2024
    assert next(d for d in docs if d.document_id == "C1").metadata.title == "CSV title"
    assert report.records_skipped >= 2  # bad JSON line + empty text
    assert report.files_skipped >= 1  # unknown .bin
    assert all(d.metadata.attributes["source"] == "s" for d in docs)
    assert all("\r" not in d.text for d in docs)


def test_document_ids_do_not_depend_on_absolute_location(tmp_path: Path) -> None:
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "x.txt").write_text("hello world", encoding="utf-8")
    a = list(iter_corpus(_loaded(tmp_path, [{"id": "s", "path": str(tmp_path / "one"), "id_field": None}])))
    b = list(iter_corpus(_loaded(tmp_path, [{"id": "s", "path": str(tmp_path / "two"), "id_field": None}])))
    assert a[0].document_id == b[0].document_id
    assert a[0].source.source_key == "x.txt"


def test_ingestion_limits_and_safety(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    (root / "big.txt").write_text("x" * 5000, encoding="utf-8")
    (root / "ok.txt").write_text("fine", encoding="utf-8")
    (root / "bad.txt").write_bytes(b"caf\xe9 latin1")
    report = IngestionReport()
    files = list(iter_source_files(root, "*.txt", IngestionLimits(max_file_bytes=1000), report))
    assert {f.name for f in files} == {"ok.txt", "bad.txt"}
    with pytest.raises(UnsafePathError):
        list(iter_source_files(root, "../*.txt", IngestionLimits(), report))
    with pytest.raises(CorpusNotFound):
        list(iter_source_files(tmp_path / "missing", "*", IngestionLimits(), report))
    with pytest.raises(IngestionError):
        decode_bytes(b"\xff\xfe\xfa", IngestionLimits(on_decode_error="fail"), report, "x")
    assert decode_bytes(b"\xffok", IngestionLimits(on_decode_error="skip"), report, "x") is None
    assert "ok" in (decode_bytes(b"\xffok", IngestionLimits(), report, "x") or "")


def test_duplicate_ids_first_wins_and_max_documents(tmp_path: Path) -> None:
    f = tmp_path / "d.jsonl"
    f.write_text(
        '{"id": "X", "text": "first"}\n{"id": "X", "text": "second"}\n{"id": "Y", "text": "y"}\n', encoding="utf-8"
    )
    report = IngestionReport()
    docs = list(iter_corpus(_loaded(tmp_path, [{"id": "s", "path": str(f)}]), report))
    assert [d.text for d in docs] == ["first", "y"] and report.duplicate_ids == 1
    capped = list(iter_corpus(_loaded(tmp_path, [{"id": "s", "path": str(f)}], max_documents=1)))
    assert len(capped) == 1


def _doc(text: str) -> Document:
    return Document(
        document_id="D",
        text=text,
        source=DocumentSource("s", "k", SourceFormat.TEXT),
        metadata=DocumentMetadata(title="T", attributes={"category": "ops"}),
        content_hash="h",
    )


LONG = (
    "Intro paragraph sentence one. Sentence two follows here!\n\n# Guide\n\nThe guide starts. "
    + " ".join(f"Step {i} explains a detail of the system in a full sentence." for i in range(30))
    + "\n\n## Rollback\n\nRoll back with the CLI. It is fast.\n\n## Tiny\n\nok\n"
)


@pytest.mark.parametrize("strategy", list(ChunkStrategy))
def test_chunk_offsets_determinism_and_metadata(strategy: ChunkStrategy) -> None:
    cfg = ChunkingConfig(
        strategy=strategy, max_chars=300, overlap_chars=60, min_chars=20, max_tokens=40, overlap_tokens=10
    )
    chunker = build_chunker(cfg)
    doc = _doc(LONG)
    chunks = list(chunker.chunk(doc))
    assert chunks and [c.sequence for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert doc.text[c.metadata.start_char : c.metadata.end_char] == c.text
        assert c.text == c.text.strip() and c.metadata.attributes == {"category": "ops"}
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in chunker.chunk(doc)]
    covered = set()
    for c in chunks:
        covered.update(range(c.metadata.start_char, c.metadata.end_char))
    assert all(i in covered for i, ch in enumerate(doc.text) if not ch.isspace())  # no text lost


def test_sentence_overlap_and_window_limit() -> None:
    chunks = list(
        SentenceChunker(
            ChunkingConfig(strategy=ChunkStrategy.SENTENCE, max_chars=200, overlap_chars=80, min_chars=10)
        ).chunk(_doc(LONG))
    )
    assert len(chunks) > 3
    overlaps = sum(1 for a, b in itertools.pairwise(chunks) if b.metadata.start_char < a.metadata.end_char)
    assert overlaps >= 1
    assert all(len(c.text) <= 200 for c in chunks)
    none = list(
        SentenceChunker(
            ChunkingConfig(strategy=ChunkStrategy.SENTENCE, max_chars=200, overlap_chars=0, min_chars=10)
        ).chunk(_doc(LONG))
    )
    assert all(b.metadata.start_char >= a.metadata.end_char for a, b in itertools.pairwise(none))


def test_fixed_token_windows() -> None:
    text = " ".join(f"w{i}" for i in range(100))
    chunks = list(
        FixedTokenApproxChunker(
            ChunkingConfig(strategy=ChunkStrategy.FIXED_TOKEN, max_tokens=30, overlap_tokens=10, min_chars=1)
        ).chunk(_doc(text))
    )
    assert [len(c.text.split()) for c in chunks][:4] == [30, 30, 30, 30]
    assert chunks[1].text.split()[0] == "w20"


def test_section_paths_and_short_merge() -> None:
    chunks = list(SectionChunker(ChunkingConfig(max_chars=2000, overlap_chars=0, min_chars=20)).chunk(_doc(LONG)))
    sections = [c.metadata.section for c in chunks]
    assert "Guide > Rollback" in sections
    assert not any(c.text.strip() == "## Tiny\n\nok" for c in chunks)  # merged, not dropped
    assert "ok" in chunks[-1].text


def test_empty_and_merge_helpers() -> None:
    assert list(build_chunker(ChunkingConfig()).chunk(_doc("   "))) == []
    merged = merge_short_spans([Span(0, 5), Span(6, 100), Span(101, 103)], 10)
    assert merged == [Span(0, 103)]
    with pytest.raises(ValueError, match="overlap"):
        ChunkingConfig(max_chars=100, overlap_chars=100)
