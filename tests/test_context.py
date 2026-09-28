# harness/context.py: image pruning of a chat history between tool
# rounds. The rules that matter to a running harness: text always
# survives, the newest images survive, the caller's own history is never
# mutated (it keeps the full-fidelity copy), and non-image messages are
# passed through untouched, identity included.
import copy

import pytest

from harness.context import prune_context_images


def _img(label, kind="image"):
    block = {"type": "image", "source": {"data": f"<{label}>"}} if kind == "image" \
        else {"type": "image_url", "image_url": {"url": f"data:{label}"}}
    return {"role": "user", "content": [{"type": "text", "text": f"caption {label}"}, block]}


def _text(role, text):
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _has_image(msg):
    return any(b["type"] in ("image", "image_url") for b in msg["content"])


def test_keeps_only_last_n_images_and_all_text():
    history = [_text("user", "go"), _img("f1"), _text("assistant", "ok"), _img("f2"), _img("f3"), _img("f4")]
    pruned = prune_context_images(history, keep_last_n=2)
    assert [_has_image(m) for m in pruned] == [False, False, False, False, True, True]
    assert pruned[1]["content"] == [{"type": "text", "text": "caption f1"}]
    assert pruned[3]["content"][0]["text"] == "caption f2"


def test_keep_first_and_last_counts_image_messages_only():
    history = [_text("user", "a"), _img("f1"), _text("user", "b"), _img("f2"), _img("f3"), _text("user", "c")]
    pruned = prune_context_images(history, keep_first_n=1, keep_last_n=1)
    assert _has_image(pruned[1]) and _has_image(pruned[4])
    assert not _has_image(pruned[3])


def test_does_not_mutate_caller_history():
    history = [_img("f1"), _img("f2"), _img("f3")]
    snapshot = copy.deepcopy(history)
    prune_context_images(history, keep_last_n=1)
    assert history == snapshot


def test_non_image_messages_pass_through_by_identity():
    plain = _text("assistant", "thinking")
    history = [_img("f1"), plain, _img("f2")]
    pruned = prune_context_images(history, keep_last_n=1)
    assert pruned[1] is plain


def test_string_content_and_openai_image_blocks():
    history = [{"role": "user", "content": "plain string"}, _img("f1", kind="image_url"), _img("f2", kind="image_url")]
    pruned = prune_context_images(history, keep_last_n=1)
    assert pruned[0] == {"role": "user", "content": "plain string"}
    assert not _has_image(pruned[1]) and _has_image(pruned[2])


def test_zero_keeps_strips_everything_and_defaults_are_zero():
    history = [_img("f1"), _img("f2")]
    assert not any(_has_image(m) for m in prune_context_images(history))


def test_keep_more_than_exist_is_a_no_op():
    history = [_img("f1"), _img("f2")]
    assert prune_context_images(history, keep_last_n=10) == history


def test_negative_counts_rejected():
    with pytest.raises(ValueError):
        prune_context_images([], keep_last_n=-1)
    with pytest.raises(ValueError):
        prune_context_images([], keep_first_n=-1)
