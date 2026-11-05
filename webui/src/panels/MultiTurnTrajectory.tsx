import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";

export interface MultiTurnData {
  enabled: boolean;
  valid: boolean;
  task_profile: string;
  test_metric?: string;
  final_reward?: number;
  training_reward?: number;
  num_user_rounds?: number;
  stop_reason?: string;
  error?: string;
  round_rewards?: { round_id: number; functional_reward: number; judge_score?: number; training_reward: number }[];
  rounds?: { round_id: number; instruction: string; status?: string; final_message?: string;
    functional_reward?: number; judge_score?: number; training_reward?: number;
    judge_result?: { rationale?: string }; test_output?: string; agent_trace?: string }[];
}

export default function MultiTurnTrajectory({ value }: { value?: MultiTurnData | null }) {
  if (!value?.enabled) return null;
  const rows = value.round_rewards ?? [];
  const last = rows[rows.length - 1];
  const detailRows: NonNullable<MultiTurnData["rounds"]> = value.rounds ?? rows.map(r => ({ ...r, instruction: "" }));
  const fmt = (v?: number) => v == null ? "—" : v.toFixed(3);
  return <div className="rounded-xl border border-slate-700 p-4 my-4 text-slate-300 space-y-3">
    <h3 className="font-semibold">Multi-turn · {value.task_profile}</h3>
    <p className="text-xs">Final reward: {value.valid ? fmt(value.training_reward ?? value.final_reward ?? last?.training_reward) : "invalid"} · Feedbacks: {Math.max(0, (value.num_user_rounds ?? value.rounds?.length ?? rows.length) - 1)} · Stop: {value.stop_reason ?? value.error ?? "unknown"}</p>
    {rows.length > 0 && <ResponsiveContainer width="100%" height={220}>
      <LineChart data={rows}><XAxis dataKey="round_id" label={{ value: "User round", position: "insideBottom", offset: -2 }} /><YAxis domain={[0, 1]} /><Tooltip /><Legend />
        <Line dataKey="functional_reward" name={value.task_profile === "openswe" ? "Test success (0/1)" : "Test score"} stroke="#199e70" />
        <Line dataKey="judge_score" name="Judge" stroke="#7a6ddb" />
        <Line dataKey="training_reward" name="Combined reward" stroke="#e0a01a" />
      </LineChart>
    </ResponsiveContainer>}
    {!value.valid && <p className="text-xs text-amber-400">Partial round results are diagnostic only; this episode is invalid for training.</p>}
    {detailRows.map(r => <details key={r.round_id} className="border border-slate-800 rounded p-3">
      <summary className="cursor-pointer text-xs">Round {r.round_id} · Test {fmt(r.functional_reward)} · Judge {fmt(r.judge_score)} · Reward {fmt(r.training_reward)}</summary>
      {Object.entries({ "User": r.instruction, "Agent answer": "final_message" in r ? r.final_message : "", "Judge rationale": "judge_result" in r ? r.judge_result?.rationale : "", "Test output (tail)": "test_output" in r ? r.test_output : "", "Agent/tool trace (tail)": "agent_trace" in r ? r.agent_trace : "" }).map(([label, text]) => text ? <div key={label} className="mt-3"><h4 className="text-xs text-slate-500">{label}</h4><pre className="text-xs whitespace-pre-wrap break-words max-h-64 overflow-auto">{text}</pre></div> : null)}
    </details>)}
  </div>;
}
