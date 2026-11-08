import numpy as np
import pytest
import torch
from verl import DataProto
from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    _align_optional_agent_loop_timings,
    addition_process,
    assemble_batch_from_rollout_samples,
)


def test_optional_timing_missing_from_one_of_32_rollouts():
    for missing_index in (0, 17):
        batches = []
        for i in range(32):
            metrics = {"generate_sequences": 1.0, "tool_calls": 0.0}
            if i != missing_index:
                metrics.update(
                    r3_generated_tokens=float(i + 1),
                    r3_missing_generated_routes=0.0,
                    r3_payload_bytes=0.0,
                )
            batch = DataProto.from_single_dict({"input_ids": torch.tensor([[i]])})
            batch.meta_info["metrics"] = [metrics]
            batches.append(addition_process(batch))

        _align_optional_agent_loop_timings(batches)
        combined = DataProto.concat(batches)

        values = combined.non_tensor_batch["_al_timing_r3_generated_tokens"]
        assert len(values) == 32
        np.testing.assert_array_equal(
            values, [0.0 if i == missing_index else i + 1 for i in range(32)]
        )
        for key in ("r3_missing_generated_routes", "r3_payload_bytes"):
            assert len(combined.non_tensor_batch[f"_al_timing_{key}"]) == 32


def test_optional_timing_absent_from_every_rollout_stays_absent():
    batches = []
    for i in range(2):
        batch = DataProto.from_single_dict({"input_ids": torch.tensor([[i]])})
        batch.meta_info["metrics"] = [{"generate_sequences": 1.0, "tool_calls": 0.0}]
        batches.append(addition_process(batch))

    _align_optional_agent_loop_timings(batches)
    combined = DataProto.concat(batches)
    assert not any(key.startswith("_al_timing_") for key in combined.non_tensor_batch)


def test_optional_timing_alignment_preserves_multirow_order_and_values():
    batches = []
    for ids, timing in [([1, 2], None), ([3, 4, 5], [7.0, 8.0, 9.0])]:
        batch = DataProto.from_single_dict(
            {"input_ids": torch.tensor(ids).reshape(-1, 1)}
        )
        if timing is not None:
            batch.non_tensor_batch["_al_timing_r3_generated_tokens"] = np.asarray(
                timing
            )
        batches.append(batch)
    _align_optional_agent_loop_timings(batches)
    _align_optional_agent_loop_timings(batches)
    combined = DataProto.concat(batches)
    np.testing.assert_array_equal(
        combined.non_tensor_batch["_al_timing_r3_generated_tokens"],
        [0.0, 0.0, 7.0, 8.0, 9.0],
    )
    assert combined.batch["input_ids"].flatten().tolist() == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("missing_index", [0, 1])
def test_assembly_aligns_context_metric_for_empty_trajectory(missing_index):
    samples = []
    for i in range(2):
        metrics = {"generate_sequences": 1.0, "tool_calls": 0.0}
        if i != missing_index:
            metrics["context_tokens_relocated"] = 12.0
        batch = DataProto.from_single_dict(
            {
                "input_ids": torch.tensor([[i]]),
                "attention_mask": torch.ones((1, 1), dtype=torch.long),
                "response_mask": torch.ones((1, 1), dtype=torch.long),
                "min_global_steps": np.array([0]),
                "max_global_steps": np.array([0]),
            }
        )
        batch.meta_info["metrics"] = [metrics]
        samples.append(RolloutSample(batch, str(i), 0, {}))

    # Exercise the production caller: omitting its alignment call must fail.
    combined = assemble_batch_from_rollout_samples(samples, tokenizer=None, config=None)

    assert combined.batch["input_ids"].flatten().tolist() == [0, 1]
    prefix = "timing_s/agent_loop/context_tokens_relocated/"
    assert combined.meta_info[prefix + "min"] == 0.0
    assert combined.meta_info[prefix + "max"] == 12.0
    assert combined.meta_info[prefix + "mean"] == 6.0
    assert not any(key.startswith("_al_timing_") for key in combined.non_tensor_batch)
