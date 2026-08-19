# pxGPT — session notes

**Read [`HANDOFF.md`](HANDOFF.md) before working on Stage 3 / OpenAI / the
shard-budget benchmarks.** It carries the conclusions already measured, so you do
not have to re-derive or re-run them.

Rules that must hold regardless of the task:

- **Never write to `../02_mature_v1/shard_master_schema/` or
  `../02_mature_v1/Result_Stage3/`.** Both are frozen (`chmod -w`, sha256 recorded)
  with a manual evaluation running against them. The only sanctioned writer is an
  explicit `--allow-reshard`.
- **After any OpenAI run, `pxgpt cleanup-files`.** The Files API bills for storage;
  confirm with `client.files.list()` returning 0.
- **`.env` is never auto-loaded.** pxGPT reads the process environment only:
  `set -a && source project_A.env && set +a`. Plain `source` does not export.
- Quantitative traits are rough ruler/colorchecker estimates from multiple angles.
  Their run-to-run inconsistency is **expected and accepted** — never fold them
  into a trait-agreement rate.
- No drive-by changes: do what was asked, then report anything else you found.
- `03_mature_v2/` is **usable** as of 2026-08-19: the correct later-stage images
  are in place (277 lines / 5208 files) and `master_schema.json` reproduces the
  frozen shard set byte-identically. Its numbers do NOT carry over from
  `02_mature_v1` — different schema, 50 traits / 13 groups / 7 quantitative
  against v1's 49 / 12 / 4, only 4 trait names shared. See `HANDOFF.md`.
- `03_mature_v2/images/` and `03_mature_v2/shard_master_schema/` are read-only
  (`dr-xr-x---`). Treat them like the v1 frozen set: never write there.
- The local vLLM server reaches images through `file://`, gated by `MEDIA_ROOT`
  in `ops/local-vllm/.env` — it is both the bind mount and
  `--allowed-local-media-path`. It is set to `/home/xavier/project/pxgpt`, so
  both datasets resolve. A path outside that tree fails with a 400; changing it
  needs `down.sh` + `up.sh`. pxGPT itself never reads `MEDIA_ROOT`.
