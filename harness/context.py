# context.py
# ------------------------------------------------------------
# Image-pruning for an LLM harness's chat history: strip image content
# from all but the most recent (and, optionally, first) N image-bearing
# messages, keeping their text - so a loop that calls get_image()
# repeatedly doesn't keep resending every frame it has ever captured on
# every subsequent model call.
#
# MESSAGE SCHEMA THIS OPERATES ON: a list of {"role": ..., "content":
# [...]} dicts (the shape used by both the OpenAI and Anthropic message
# APIs) - "content" is a list of blocks, each a dict with a "type" key.
# A block is treated as an image if its type is "image" (Anthropic-
# style) or "image_url" (OpenAI-style). Any harness loop built on top of
# this repo's MCP tools is responsible for turning get_image()'s
# [metadata, MCPImage] tool result into blocks of this shape before
# appending them to its own message history - this module doesn't know
# about MCP or a specific LLM SDK, only this message shape, so it has no
# new dependencies.
#
# Run directly for a quick sanity check, no dependencies required:
#   python -m harness.context
# ------------------------------------------------------------

from typing import Any

IMAGE_BLOCK_TYPES = {"image", "image_url"}


def _message_has_image(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") in IMAGE_BLOCK_TYPES
        for block in content
    )


def _strip_images(message: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `message` with any image blocks removed from its
    content, keeping every other block (text, tool-use, etc.) as-is.
    """
    content = message.get("content")
    if not isinstance(content, list):
        return dict(message)
    kept = [
        block for block in content
        if not (isinstance(block, dict) and block.get("type") in IMAGE_BLOCK_TYPES)
    ]
    return {**message, "content": kept}


def prune_context_images(
    messages: list[dict[str, Any]],
    keep_first_n: int = 0,
    keep_last_n: int = 0,
) -> list[dict[str, Any]]:
    """Return a new message list where image blocks have been stripped
    from every image-bearing message except the first `keep_first_n` and
    last `keep_last_n` (counted among image-bearing messages only, not
    all messages) - those are returned unchanged, images included.
    Non-image-bearing messages always pass through unchanged.

    Does not mutate `messages` or any message dict in it - always
    returns new message/content lists, so a caller can keep its own
    full-fidelity history separately (e.g. for logging or the FrameStore
    pattern in mcp_server/loop_tools.py's get_frame()) while only
    passing the pruned copy to the model.

    Raises ValueError if either count is negative.
    """
    if keep_first_n < 0 or keep_last_n < 0:
        raise ValueError("keep_first_n and keep_last_n must be non-negative.")

    image_positions = [i for i, message in enumerate(messages) if _message_has_image(message)]
    n_images = len(image_positions)
    keep_indices = set(image_positions[:keep_first_n]) | set(image_positions[n_images - keep_last_n:])

    return [
        _strip_images(message) if index in image_positions and index not in keep_indices else message
        for index, message in enumerate(messages)
    ]


if __name__ == "__main__":
    def _image_message(role: str, label: str) -> dict:
        return {
            "role": role,
            "content": [
                {"type": "text", "text": f"caption for {label}"},
                {"type": "image", "data": f"<{label}-bytes>"},
            ],
        }

    history = [
        {"role": "user", "content": [{"type": "text", "text": "move to sample edge"}]},
        _image_message("tool", "frame_1"),
        {"role": "assistant", "content": [{"type": "text", "text": "looks clean, moving on"}]},
        _image_message("tool", "frame_2"),
        _image_message("tool", "frame_3"),
        _image_message("tool", "frame_4"),
    ]

    print("Before pruning:")
    for message in history:
        print(" ", message)

    pruned = prune_context_images(history, keep_first_n=1, keep_last_n=1)

    print("\nAfter prune_context_images(keep_first_n=1, keep_last_n=1):")
    for message in pruned:
        print(" ", message)

    assert any(b["type"] == "image" for b in pruned[1]["content"]), "frame_1 (first) should keep its image"
    assert not any(b["type"] == "image" for b in pruned[3]["content"]), "frame_2 (middle) should lose its image"
    assert not any(b["type"] == "image" for b in pruned[4]["content"]), "frame_3 (middle) should lose its image"
    assert any(b["type"] == "image" for b in pruned[5]["content"]), "frame_4 (last) should keep its image"
    assert pruned[3]["content"][0]["text"] == "caption for frame_2", "text must survive pruning"
    assert history[3]["content"][1]["type"] == "image", "original history must not be mutated"

    print("\nAll sanity checks passed.")
