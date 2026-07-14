# Stage 3 sharded dispatch: `batch` vs `sequential`

`pxgpt phenotype-batch --shard-dir ... --dispatch {batch,sequential}` builds the
same *(plant × shard)* requests in both modes. The mode changes how requests are
sent. It does not change how Anthropic Structured Outputs affect prompt caching.

## Bottom line

- `batch` sends one asynchronous Message Batch. It has the Batch API discount
  and higher throughput.
- `sequential` sends one synchronous call at a time, with all shards for one
  plant kept together. It has no batch discount, but it provides incremental
  output writes, resume, bounded retry and live progress.
- Images are intentionally outside the cached prefix. They are ordinary input in
  both dispatch modes and stay before the per-shard text prompt.

Choose the mode for transport, throughput and recovery behavior. Prompt caching
applies only to the smaller system/format prefix.

## Request structure

`build_sharded_requests` creates requests in plant-contiguous order:

```text
plant A, shard 01
plant A, shard 02
...
plant B, shard 01
plant B, shard 02
...
```

Each request has one explicit cache breakpoint on the shared system block.

The visible request content is:

```text
system: [shared system | cache breakpoint]
user:   [plant images] [per-shard text prompt]
```

Images do not carry `cache_control`. They remain at the start of the user content,
before the text prompt, following Anthropic's recommended image-then-text layout.
The request also contains a different `output_config.format` schema for every
shard.

## Structured Outputs changes the effective cache identity

Anthropic Structured Outputs add format-specific system instructions to the
effective prompt. Changing `output_config.format` invalidates the related prompt
cache. A schema being a separate top-level API parameter does not keep it outside
the effective cache identity.

This has two important effects in sharded Stage 3:

- Across shards of the **same plant**, the schema changes, so those shards do not
  share one system/format cache identity.
- Across plants using the **same shard**, the schema stays the same, so the
  smaller system/format prefix may be read from cache.

Images remain ordinary input in both cases. This avoids paying the cache-write
premium repeatedly for image tokens that different shard schemas cannot reuse.

Structured Outputs also cache the compiled grammar separately. That grammar
cache reduces later schema-compilation latency. It is not prompt caching and
does not reduce image input tokens.

See Anthropic's official documentation for
[Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
and [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

## `--dispatch sequential`

Sequential dispatch uses a plain synchronous loop. One request must return before
the next request is sent.

It provides:

- plant-contiguous request order;
- immediate writes to `<output>/_partial/`;
- automatic resume from valid partial files;
- bounded retry for transient API errors;
- live per-request cache usage in stdout.

Serial order removes the concurrent cold-start race between adjacent requests.
It does not bypass cache invalidation caused by changing
`output_config.format`. The default five-minute cache lifetime can also matter
after long delays, errors or resumed runs.

Sequential calls use normal Messages API pricing. They do not receive the Batch
API discount.

## `--dispatch batch`

Batch dispatch submits all *(plant × shard)* requests to one asynchronous Message
Batch.

It provides:

- the Batch API discount;
- higher throughput;
- fire-and-forget submission with a checkpoint;
- later retrieval and merge through `fetch-results`.

Batch requests may run concurrently or far apart in time, so system/format cache
hits are best-effort. Structured Outputs cache invalidation still applies when
the shard schema changes. Images are ordinary input regardless of scheduling.

## Reading token usage

With images outside the cache breakpoint, expect this general pattern:

```text
input_tokens   include the plant images and per-shard text prompt
cache_read     may be stable for the same shard across different plants
cache_creation represents a cold system/format prefix, not the plant images
```

The CLI prints token counts, not the number of cache operations:

```text
input=<input_tokens>
cache_read=<cache_read_input_tokens>
cache_creation=<cache_creation_input_tokens>
```

## Practical mode selection

Use `sequential` when recovery and observability matter most. It is the safer
choice for long HPC jobs because completed shards are persisted immediately and
can be resumed without re-billing.

Use `batch` when throughput and the Batch API discount matter most, and delayed
result retrieval is acceptable.

For cost decisions, run a small representative pilot and compare actual input,
output, cache-read and cache-creation tokens. Image tokens should now appear as
ordinary input instead of a large cache creation on every shard.
