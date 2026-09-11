# Raw observation archive

`samples.jsonl` is 120,094,186 bytes, above GitHub's single-file limit. Its lossless
gzip archive `samples.jsonl.gz` is committed instead (37,666,421 bytes). The raw
file remains present locally and on A6000; it was not deleted. Decompression was
verified byte-for-byte. SHA256 hashes and byte counts are in samples_archive.json.

After a fresh clone, run from this study directory:

```sh
python -m gzip -d samples.jsonl.gz
```

Then run from the repository root:

```sh
python -m scripts.report_t003 research_log/remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study
```

`samples.jsonl` is ignored only at this exact run path. All 72 prediction files,
metrics, environment, subset, baseline audit, analysis JSON/Markdown, figures,
completion receipt and raw archive remain tracked.
