"""CPU regressions for literal vision tokens and failed async rollouts."""

import asyncio
from functools import partial
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter


IMAGE_TOKEN = 151655
VIDEO_TOKEN = 151656


@pytest.fixture
def worker():
    # Use the installed Transformers RoPE implementation without loading weights.
    processor = SimpleNamespace(
        image_token_id=IMAGE_TOKEN,
        video_token_id=VIDEO_TOKEN,
        config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)),
    )
    processor.get_rope_index = MethodType(Qwen3VLModel.get_rope_index, processor)
    processor.get_vision_position_ids = MethodType(Qwen3VLModel.get_vision_position_ids, processor)
    return SimpleNamespace(processor=processor)


@pytest.mark.parametrize("empty_grids", [False, True])
def test_literal_vision_tokens_are_text_without_media(worker, empty_grids):
    # Source code read by OC contained image_token="<|image_pad|>". Also
    # cover video literals and padding; none of these IDs should be rewritten.
    ids = torch.tensor([[0, 42, IMAGE_TOKEN, 43, VIDEO_TOKEN, 44, 0]])
    original = ids.clone()
    mask = torch.tensor([[0, 1, 1, 1, 1, 1, 0]])
    media = {"mm_token_type_ids": torch.zeros_like(ids)}
    if empty_grids:
        media.update(
            image_grid_thw=torch.empty(0, 3, dtype=torch.long),
            video_grid_thw=torch.empty(0, 3, dtype=torch.long),
        )
    positions = AgentLoopWorker._compute_position_ids(worker, ids, mask, media)
    assert positions.shape == (1, 4, 7)
    torch.testing.assert_close(positions[0, :, 1:6], torch.arange(5).expand(4, -1))
    torch.testing.assert_close(ids, original)


@pytest.mark.parametrize("modality", ["image", "video"])
def test_real_media_keeps_spatial_rope_with_other_modality_literal(worker, modality):
    token = IMAGE_TOKEN if modality == "image" else VIDEO_TOKEN
    literal = VIDEO_TOKEN if modality == "image" else IMAGE_TOKEN
    ids = torch.tensor([[42, token, token, token, token, literal, 43]])
    media = {
        "mm_token_type_ids": torch.zeros_like(ids),
        f"{modality}_grid_thw": torch.tensor([[1, 4, 4]]),
    }
    positions = AgentLoopWorker._compute_position_ids(worker, ids, torch.ones_like(ids), media)
    expected = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6],  # text positions
        [0, 1, 1, 1, 1, 3, 4],  # temporal positions
        [0, 1, 1, 2, 2, 3, 4],  # height positions
        [0, 1, 2, 1, 2, 3, 4],  # width positions
    ])
    torch.testing.assert_close(positions[0], expected)


def test_text_only_model_keeps_single_position_channel():
    ids = torch.tensor([[42, IMAGE_TOKEN, VIDEO_TOKEN]])
    positions = AgentLoopWorker._compute_position_ids(
        SimpleNamespace(processor=None), ids, torch.ones_like(ids), {}
    )
    torch.testing.assert_close(positions, torch.tensor([[0, 1, 2]]))


# Ray exposes its implementation class for local CPU testing, without starting Ray.
Rollouter = FullyAsyncRollouter.__ray_metadata__.modified_class


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_source", ["generation", "monitor"])
async def test_fit_propagates_failure_and_cancels_sibling(failure_source, capsys):
    sibling_started = asyncio.Event()
    sibling_stopped = asyncio.Event()
    error = TypeError("sample_0_2477: missing image grid")

    async def fail():
        await sibling_started.wait()
        raise error

    async def wait():
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_stopped.set()

    state = SimpleNamespace(
        lock=asyncio.Lock(), _resume_event=asyncio.Event(), message_queue_client=object(),
        _streaming_generation_main=fail if failure_source == "generation" else wait,
        _async_monitor_loop=fail if failure_source == "monitor" else wait,
    )
    with pytest.raises(TypeError) as caught:
        await asyncio.wait_for(Rollouter.fit(state), timeout=2)
    assert caught.value is error
    assert sibling_stopped.is_set()
    assert "Rollouter fit completed" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_processor_error_reaches_fit_caller(capsys):
    feed_started = asyncio.Event()
    feed_stopped = asyncio.Event()

    async def feed():
        feed_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            feed_stopped.set()

    async def processor():
        await feed_started.wait()
        raise TypeError("sample_0_2477: missing image grid")

    async def monitor():
        await asyncio.Event().wait()

    state = SimpleNamespace(
        lock=asyncio.Lock(), _resume_event=asyncio.Event(),
        message_queue_client=SimpleNamespace(put_sample=AsyncMock()),
        async_rollout_manager=object(), max_concurrent_samples=1,
        _feed_samples=feed, _processor_worker=processor, _async_monitor_loop=monitor,
    )
    state._streaming_generation_main = partial(Rollouter._streaming_generation_main, state)
    with pytest.raises(TypeError, match="sample_0_2477"):
        await asyncio.wait_for(Rollouter.fit(state), timeout=2)
    assert feed_stopped.is_set()
    assert state.running is False
    assert state.feed_task is None and state.processor_task is None
    state.message_queue_client.put_sample.assert_awaited_once_with(sample=None)
    assert "Rollouter fit completed" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_fit_normal_completion(capsys):
    finished = asyncio.Event()

    async def generate():
        finished.set()

    state = SimpleNamespace(
        lock=asyncio.Lock(), _resume_event=asyncio.Event(), message_queue_client=object(),
        _streaming_generation_main=generate, _async_monitor_loop=finished.wait,
    )
    await asyncio.wait_for(Rollouter.fit(state), timeout=2)
    assert "Rollouter fit completed" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_fit_cancellation_cleans_up_both_tasks():
    started = [asyncio.Event(), asyncio.Event()]
    stopped = [asyncio.Event(), asyncio.Event()]

    async def wait(index):
        started[index].set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped[index].set()

    state = SimpleNamespace(
        lock=asyncio.Lock(), _resume_event=asyncio.Event(), message_queue_client=object(),
        _streaming_generation_main=partial(wait, 0), _async_monitor_loop=partial(wait, 1),
    )
    task = asyncio.create_task(Rollouter.fit(state))
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), timeout=2)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert all(event.is_set() for event in stopped)
