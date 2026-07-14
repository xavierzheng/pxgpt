"""Regression tests for Stage 3 sharded request cache layout."""

from pxgpt.core.sharding import build_sharded_requests


class _Config:
    anthropic_model = "claude-sonnet-test"
    stage3_max_tokens = 100
    temperature = 0

    @staticmethod
    def stage3_output_config(schema):
        return {"format": {"type": "json_schema", "schema": schema}}


def test_sharded_requests_cache_only_system_and_keep_images_before_prompt():
    image_blocks = [
        {"type": "image", "source": {"type": "file", "file_id": "file_1"}},
        {"type": "image", "source": {"type": "file", "file_id": "file_2"}},
    ]
    shards = [
        {"shard_id": "shard_01", "prompt": "Score shard 1.", "schema": {"type": "object"}},
        {"shard_id": "shard_02", "prompt": "Score shard 2.", "schema": {"type": "object"}},
    ]

    requests = build_sharded_requests(
        {"plant_1": image_blocks},
        shards,
        "Shared system prompt.",
        _Config(),
        lambda **kwargs: kwargs,
    )

    assert len(requests) == 2
    for request, shard in zip(requests, shards):
        params = request["params"]
        assert params["system"] == [{
            "type": "text",
            "text": "Shared system prompt.",
            "cache_control": {"type": "ephemeral"},
        }]
        assert params["messages"][0]["content"] == [
            *image_blocks,
            {"type": "text", "text": shard["prompt"]},
        ]

    assert all("cache_control" not in block for block in image_blocks)
