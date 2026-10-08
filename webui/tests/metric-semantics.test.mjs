import test from "node:test";
import assert from "node:assert/strict";
import { withRewardMetrics } from "../src/utils/rewardMetrics.ts";
import { withTrainingStep, trainingStepCount } from "../src/utils/trainingSteps.ts";

test("explicit episode and segment statistics remain different; raw metrics unchanged", () => {
  const raw = { step: 8, "reward_stats/schema_version": 2, "reward_stats/trial_mean": 0.5,
    "reward_stats/segment_mean": 0.75, "reward_stats/padding_ratio": 0.2, "critic/rewards/mean": 0.6 };
  const result = withRewardMetrics(raw);
  assert.equal(result["dashboard/reward/trial_mean"], 0.5);
  assert.equal(result["dashboard/reward/segment_mean"], 0.75);
  assert.equal(result["dashboard/reward/padding_ratio"], 0.2);
  for (const key of Object.keys(raw)) assert.equal(result[key], raw[key]);
  assert.equal(raw["dashboard/reward/trial_mean"], undefined);
});
const legacy = { step: 1, "timing_s/agent_loop/harness_reward/cc/mean": 0.25,
  "timing_s/agent_loop/harness_reward/codex/mean": 0.5,
  "timing_s/agent_loop/harness_selected/cc/mean": 0.5,
  "timing_s/agent_loop/harness_selected/codex/mean": 0.5,
  "timing_s/agent_loop/proxy_num_segments/min": 1,
  "timing_s/agent_loop/proxy_num_segments/max": 1,
  "timing_s/agent_loop/proxy_num_compactions/max": 0 };
test("proven single-segment legacy means are summed, not averaged", () => {
  assert.equal(withRewardMetrics(legacy)["dashboard/reward/trial_mean"], 0.75);
});
test("ambiguous legacy logs and unknown schemas cannot manufacture trial means", () => {
  for (const update of [
    { "timing_s/agent_loop/proxy_num_segments/max": 3 },
    { "timing_s/agent_loop/proxy_num_compactions/max": 1 },
    { "timing_s/agent_loop/harness_selected/cc/mean": 0.1 },
    { "timing_s/agent_loop/harness_selected/cc/mean": undefined },
    { "reward_stats/schema_version": 3 },
  ]) assert.equal(withRewardMetrics({...legacy, ...update})["dashboard/reward/trial_mean"], undefined);
  assert.equal(withRewardMetrics({step: 1, "critic/score/mean": 0.9})["dashboard/reward/trial_mean"], undefined);
});
test("zero is real, missing/nonfinite fields are not zero; padding stays bounded", () => {
  assert.equal(withRewardMetrics({step: 1, "reward_stats/schema_version": 2, "reward_stats/trial_mean": 0})["dashboard/reward/trial_mean"], 0);
  assert.equal(withRewardMetrics({step: 1, "reward_stats/schema_version": 2, "reward_stats/trial_mean": NaN})["dashboard/reward/trial_mean"], undefined);
  assert.equal(withRewardMetrics({step: 1, "reward_stats/schema_version": 2, "reward_stats/padding_ratio": 2})["dashboard/reward/padding_ratio"], undefined);
  assert.equal(withRewardMetrics({step: 1, "fully_async/compaction/real_rows": 13, "fully_async/compaction/padding_rows": 3})["dashboard/reward/padding_ratio"], 3/16);
});
test("training step comes from trainer; sparse validation retains checkpoint step", () => {
  assert.equal(withTrainingStep({step: 9, "training/global_step": 8}).step, 8);
  assert.equal(withTrainingStep({step: 5, "val/score": 1}).step, 5);
  assert.equal(trainingStepCount([{step: 1}, {step: 1}, {step: 3, "training/global_step": 20}]), 20);
  assert.equal(trainingStepCount([{step: 1}, {step: 1}, {step: 3}]), 2);
  assert.equal(trainingStepCount([]), 0);
});

test("overview renders distinct trial and segment values and explicit missing values", async () => {
  const { build } = await import("esbuild");
  const { writeFile, unlink } = await import("node:fs/promises");
  const { pathToFileURL, fileURLToPath } = await import("node:url");
  const output = fileURLToPath(new URL(`./.overview-${process.pid}.mjs`, import.meta.url));
  const result = await build({
    stdin: { contents: `import React from 'react'; import {renderToStaticMarkup} from 'react-dom/server';
      import Overview from './src/panels/OverviewPanel.tsx';
      export const render = (data) => renderToStaticMarkup(React.createElement(Overview, {data}));`,
      resolveDir: fileURLToPath(new URL("..", import.meta.url)), loader: "tsx" },
    bundle: true, platform: "node", format: "esm", packages: "external", write: false,
    jsx: "automatic",
  });
  await writeFile(output, result.outputFiles[0].text);
  try {
    const { render } = await import(pathToFileURL(output).href);
    const html = render([withRewardMetrics({step: 1, "reward_stats/schema_version": 2,
      "reward_stats/trial_mean": 0.5, "reward_stats/segment_mean": 0.75})]);
    assert.match(html, /Trial Reward/);
    assert.match(html, /Segment Reward/);
    assert.match(html, /0\.5000/);
    assert.match(html, /0\.7500/);
    assert.match(render([{step: 1, "critic/score/mean": 0.9}]), /display --/);
  } finally { await unlink(output); }
});
