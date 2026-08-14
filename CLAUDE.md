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
