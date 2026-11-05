import { useState } from "react";
import ChartPanel from "../components/Chart";
import type { MetricPoint } from "../types";

export default function MultiTurnPanel({ data, prefix = "multiturn" }: { data: MetricPoint[]; prefix?: string }) {
  const [selected, setSelected] = useState("");
  const profiles = [...new Set(data.flatMap(p => Object.keys(p)
    .filter(k => k.startsWith(`${prefix}/`)).map(k => k.split("/")[1])))].sort();
  if (!profiles.length) return null;
  const profile = profiles.includes(selected) ? selected : profiles[0];
  const base = `${prefix}/${profile}`;
  const terminalOnly = profile === "swe_interact";
  const rounds = [...new Set(data.flatMap(p => Object.keys(p)
    .filter(k => k.startsWith(`${base}/round/`)).map(k => Number(k.split("/")[3]))))].sort((a, b) => a - b);
  const roundKeys = (suffix: string) => rounds.map(r => `${base}/round/${r}/${suffix}`);
  const chart = (title: string, keys: string[], description?: string) => (
    <div><ChartPanel title={title} data={data} keys={keys} />{description && <p className="text-xs text-slate-400">{description}</p>}</div>
  );
  return <div className="my-5 space-y-3">
    <div className="flex items-center gap-3">
      <h3 className="text-slate-100 font-semibold">Multi-turn · {prefix === "multiturn" ? "Training batch" : prefix === "val-multiturn" ? "Validation" : "Saved artifacts"}</h3>
      <select aria-label="Task profile" className="bg-slate-800 text-slate-200 rounded px-2 py-1 text-xs" value={profile} onChange={e => setSelected(e.target.value)}>
        {profiles.map(p => <option key={p}>{p}</option>)}
      </select>
    </div>
    <p className="text-xs text-slate-400">Final reward is the terminal score, not a sum of user-round rewards. Each episode/round counts once. Later-round means include only episodes reaching that round; cumulative success includes early successes. Invalid episodes are reported separately.</p>
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {chart("Final reward", terminalOnly ? [`${base}/final_reward`, `${base}/training_reward`] : [`${base}/final_reward`, `${base}/reward_gain`])}
      {chart("Test success", terminalOnly ? [`${base}/final_success_rate`, `${base}/success_rate_all_attempts`] : [`${base}/first_success_rate`, `${base}/final_success_rate`, `${base}/recovery_rate`])}
      {chart("Reward by user round", roundKeys("reward"))}
      {chart(profile === "openswe" ? "Binary test success by round" : "Test score by round", roundKeys("test"))}
      {data.some(p => roundKeys("judge").some(k => Number.isFinite(p[k]))) && chart("Judge correctness by round", roundKeys("judge"))}
      {chart("Cumulative test success", roundKeys("cumulative_success_rate"))}
      {chart("Observed episodes by round", roundKeys("count"))}
      {chart("Feedback count", [`${base}/feedback_count`])}
      {chart("Episode duration (seconds)", [`${base}/duration_sec`])}
      {chart("Invalid episode ratio", [`${base}/invalid_ratio`])}
      {chart("Failures by stage", ["environment", "agent", "verifier", "judge", "simulator"].map(r => `${base}/failure/${r}_ratio`))}
      {chart("Termination reasons", (terminalOnly ? ["agent_completed", "max_turns_reached", "overlong", "timeout"] : ["verifier_success", "max_rounds", "simulator_no_op", "error", "cancelled"]).map(r => `${base}/stop/${r}_ratio`))}
      {prefix === "multiturn" && <>
        {chart("Valid / trainable episodes", [`${base}/valid_episodes`, `${base}/trainable_episodes`])}
        {chart("Generated tokens per trainable episode", [`${base}/generated_tokens`])}
        {chart("Trainable episodes by round", roundKeys("trainable_count"))}
        {chart("Singleton GRPO groups", roundKeys("singleton_group_ratio"), "A task/round group with only one trainable episode receives zero relative advantage.")}
        {chart("Nonzero advantage ratio", roundKeys("nonzero_advantage_ratio"))}
        {chart("Advantage mean by round", roundKeys("advantage_mean"))}
        {chart("Advantage standard deviation", roundKeys("advantage_std"))}
        {chart("Reward standard deviation", roundKeys("reward_std"))}
      </>}
    </div>
  </div>;
}
