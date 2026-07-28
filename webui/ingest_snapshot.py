"""Ingest a colleague's rollout result bundle into a webui "snapshot run".

A bundle is one training step dumped as three files (zip or directory):
  - <name>.jsonl                  brief: {input, output, gts, score, step}
  - rollout_trajectories_*.json   full chat_completions + task metadata + token ids
  - rollout_output_*.json         verl tensor batch (input_ids/responses/... ) [optional]

The three files describe the SAME N rollouts (one step). We materialize them
into the on-disk trial-dir layout the dashboard already scans, so every existing
panel (Rollout Dist / Task Grid / Trajectory / Episode) works unchanged:

  <out>/trials/step_0001/<instance_id>-<idx6hex>/
      agent/trajectory.json        event-stream format (_parse_trajectory_json)
      verifier/reward.txt          bare reward number

Reward is taken from the .jsonl `score` joined POSITIONALLY with the trajectory
stream (verified: line i <-> trajectory i). The per-trajectory `reward` field in
rollout_trajectories is unpopulated (all 0). Group key = task.extra_args.id
(instance_id); 8 samples per prompt.
"""
from __future__ import annotations

import json
import math
import os
import re
import zipfile
from array import array


# --------------------------------------------------------------------------
# streaming json: yield each top-level object of the `trajectories` array
# --------------------------------------------------------------------------
def iter_trajectories(fp, chunk_size: int = 4 << 20):
    """fp: binary file-like positioned at start of the file. Yields dict per
    trajectory without loading the whole (156MB) file."""
    pending = ""
    started = False
    while not started:
        more = fp.read(chunk_size)
        if not more:
            return
        pending += more.decode("utf-8", "replace")
        k = pending.find('"trajectories"')
        if k != -1:
            lb = pending.find("[", k)
            if lb != -1:
                pending = pending[lb + 1 :]
                started = True

    buf = pending
    i = 0
    depth = 0
    instr = False
    esc = False
    start = None
    while True:
        while i < len(buf):
            c = buf[i]
            if esc:
                esc = False
                i += 1
                continue
            if c == "\\":
                esc = True
                i += 1
                continue
            if c == '"':
                instr = not instr
                i += 1
                continue
            if instr:
                i += 1
                continue
            if c == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield json.loads(buf[start : i + 1])
                    buf = buf[i + 1 :]
                    i = 0
                    start = None
                    continue
            elif c == "]" and depth == 0:
                return
            i += 1
        more = fp.read(chunk_size)
        if not more:
            return
        buf += more.decode("utf-8", "replace")


# --------------------------------------------------------------------------
# bundle access (zip or directory) — uniform open()/names interface
# --------------------------------------------------------------------------
class _Bundle:
    def __init__(self, path: str):
        self.path = path
        self.zip = None
        if zipfile.is_zipfile(path):
            self.zip = zipfile.ZipFile(path)
            self.names = [n for n in self.zip.namelist() if not n.endswith("/")]
        elif os.path.isdir(path):
            self.names = sorted(os.listdir(path))
        else:
            raise ValueError(f"not a zip or directory: {path}")

    def find(self, *, suffix=None, prefix=None):
        for n in self.names:
            base = os.path.basename(n)
            if suffix and not base.endswith(suffix):
                continue
            if prefix and not base.startswith(prefix):
                continue
            return n
        return None

    def open(self, name):
        if self.zip is not None:
            return self.zip.open(name)
        return open(os.path.join(self.path, name), "rb")

    def close(self):
        if self.zip is not None:
            self.zip.close()


_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe(s: str) -> str:
    return _SAFE.sub("_", s or "unknown")


def _fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# chat_completions (OpenAI format) -> event-stream trajectory.json
# --------------------------------------------------------------------------
def _chat_to_trajectory(chat: list[dict]) -> dict:
    """Map OpenAI chat messages to the {steps:[{source,message,tool_calls,
    observation}]} format that server._parse_trajectory_json reads.

    Tool results (role=tool) are paired back to the assistant turn that issued
    the matching tool_call_id."""
    # index tool results by call id
    results_by_id: dict[str, dict] = {}
    for m in chat:
        if m.get("role") == "tool":
            cid = m.get("tool_call_id")
            if cid:
                results_by_id[cid] = m

    steps: list[dict] = []
    for m in chat:
        role = m.get("role")
        if role == "system":
            steps.append({"source": "system", "message": m.get("content") or ""})
        elif role == "user":
            steps.append({"source": "user", "message": m.get("content") or ""})
        elif role == "assistant":
            tool_calls = []
            obs_results = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                tool_calls.append(
                    {
                        "function_name": fn.get("name", ""),
                        "arguments": fn.get("arguments", ""),
                    }
                )
                res = results_by_id.get(tc.get("id"))
                if res is not None:
                    obs_results.append({"content": res.get("content", "")})
            # prefer reasoning_content as the visible thought, fall back to content
            msg = m.get("content") or ""
            reasoning = m.get("reasoning_content") or ""
            thought = (reasoning + ("\n\n" + msg if msg and reasoning else "")) or msg
            steps.append(
                {
                    "source": "agent",
                    "message": thought,
                    "tool_calls": tool_calls,
                    "observation": {"results": obs_results},
                }
            )
        # role == "tool" already folded into the assistant turn
    return {"steps": steps}


# --------------------------------------------------------------------------
# token tensors (rollout_output_*.json) -> IS / logprob diagnostics
#
# verl batch dumped as {"output": {"<key>": "[[...]]"}}; each value is a JSON
# STRING of a list-of-lists (numbers only -> no quotes/backslashes inside, so a
# value ends at the next `"`). Big prompt-side fields are skipped without
# buffering; only `want` keys are parsed. The 1GB file streams in ~15s.
# --------------------------------------------------------------------------
def _stream_output_fields(fp, want, chunk_size: int = 8 << 20):
    want = set(want)
    buf = ""

    def _read():
        nonlocal buf
        d = fp.read(chunk_size)
        if not d:
            return False
        buf += d.decode("utf-8", "replace")
        return True

    while '"output"' not in buf:
        if not _read():
            return
    buf = buf[buf.index('"output"') + len('"output"'):]
    while "{" not in buf:
        if not _read():
            return
    buf = buf[buf.index("{") + 1:]
    i = 0

    def find_char(ch, start):
        nonlocal buf
        while True:
            j = buf.find(ch, start)
            if j != -1:
                return j
            buf = buf[start:]
            start = len(buf)
            if not _read():
                return -1

    while True:
        q1 = find_char('"', i)
        if q1 == -1:
            return
        close = buf.find("}", i)
        if close != -1 and close < q1:
            return
        q2 = find_char('"', q1 + 1)
        if q2 == -1:
            return
        key = buf[q1 + 1:q2]
        colon = find_char(":", q2 + 1)
        v1 = find_char('"', colon + 1)
        if key in want:
            parts = []
            start = v1 + 1
            while True:
                j = buf.find('"', start)
                if j != -1:
                    parts.append(buf[start:j])
                    i = j + 1
                    break
                parts.append(buf[start:])
                buf = ""
                start = 0
                if not _read():
                    return
            yield key, json.loads("".join(parts))
        else:
            start = v1 + 1
            while True:
                j = buf.find('"', start)
                if j != -1:
                    buf = buf[j + 1:]
                    i = 0
                    break
                buf = ""
                start = 0
                if not _read():
                    return


def _compute_token_stats(bundle: "_Bundle", oname: str, rewards, progress, n_series=160):
    """Per-token IS / logprob diagnostics from old_log_probs vs rollout_log_probs.

    IS ratio (training vs rollout policy) per response token = exp(old-roll).
    Returns a compact dict (global stats + per-sample summaries + downsampled
    per-sample log-ratio traces) safe to ship to the browser."""
    want = ["response_mask", "old_log_probs", "rollout_log_probs"]
    fields: dict[str, list] = {}
    progress(0, 1, "parsing token tensors (1GB)")
    with bundle.open(oname) as fp:
        for k, v in _stream_output_fields(fp, want):
            # compact each row immediately to bound memory
            if k == "response_mask":
                fields[k] = [array("b", [1 if x else 0 for x in row]) for row in v]
            else:
                fields[k] = [array("f", row) for row in v]
            del v
            progress(0, 1, f"parsed {k}")

    mask = fields.get("response_mask")
    olp = fields.get("old_log_probs")
    rlp = fields.get("rollout_log_probs")
    if not (mask and olp and rlp):
        return None

    nrows = min(len(mask), len(olp), len(rlp))
    # global accumulators (Welford-free running sums; tokens are ~3.7M)
    gN = gsx = gsy = gsxx = gsyy = gsxy = 0.0
    HB = 40  # histogram bins for log-ratio over [-1, 1] + 2 overflow
    lo, hi = -1.0, 1.0
    hist = [0] * (HB + 2)
    per_sample = []
    series = []

    for r in range(nrows):
        mr, ar, br = mask[r], olp[r], rlp[r]
        L = min(len(mr), len(ar), len(br))
        valid = [i for i in range(L) if mr[i]]
        n = len(valid)
        if n == 0:
            per_sample.append(
                {"idx": r, "reward": rewards[r] if r < len(rewards) else None,
                 "n_tokens": 0, "mean_old": None, "mean_roll": None,
                 "mean_log_ratio": None, "ess_frac": None, "seq_log_ratio": None,
                 "max_abs_log_ratio": None, "pearson": None}
            )
            series.append([])
            continue
        sx = sy = sxx = syy = sxy = 0.0
        sw = sw2 = slr = 0.0
        maxabs = 0.0
        lrs = []
        for i in valid:
            x = ar[i]; y = br[i]
            lr = x - y
            lrs.append(lr)
            sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y
            slr += lr
            if abs(lr) > maxabs:
                maxabs = abs(lr)
            w = math.exp(lr) if -30 < lr < 30 else (0.0 if lr <= -30 else math.exp(30))
            sw += w; sw2 += w * w
            # histogram bin
            if lr < lo:
                hist[0] += 1
            elif lr >= hi:
                hist[HB + 1] += 1
            else:
                hist[1 + int((lr - lo) / (hi - lo) * HB)] += 1
        gN += n; gsx += sx; gsy += sy; gsxx += sxx; gsyy += syy; gsxy += sxy
        cov = sxy / n - (sx / n) * (sy / n)
        vx = sxx / n - (sx / n) ** 2
        vy = syy / n - (sy / n) ** 2
        pear = cov / math.sqrt(vx * vy) if vx > 1e-12 and vy > 1e-12 else None
        ess = (sw * sw) / sw2 if sw2 > 0 else 0.0
        per_sample.append(
            {
                "idx": r,
                "reward": rewards[r] if r < len(rewards) else None,
                "n_tokens": n,
                "mean_old": round(sx / n, 4),
                "mean_roll": round(sy / n, 4),
                "mean_log_ratio": round(slr / n, 5),
                "ess_frac": round(ess / n, 4),
                "seq_log_ratio": round(slr, 3),
                "max_abs_log_ratio": round(maxabs, 3),
                "pearson": round(pear, 4) if pear is not None else None,
            }
        )
        # downsample log-ratio to n_series points (mean-pool)
        if n <= n_series:
            ds = [round(v, 4) for v in lrs]
        else:
            step = n / n_series
            ds = []
            for k in range(n_series):
                a = int(k * step); bnd = int((k + 1) * step)
                seg = lrs[a:bnd] or lrs[a:a + 1]
                ds.append(round(sum(seg) / len(seg), 4))
        series.append(ds)
        if r % 16 == 0:
            progress(r, nrows, f"IS stats {r}/{nrows}")

    gcov = gsxy / gN - (gsx / gN) * (gsy / gN)
    gvx = gsxx / gN - (gsx / gN) ** 2
    gvy = gsyy / gN - (gsy / gN) ** 2
    gpear = gcov / math.sqrt(gvx * gvy) if gvx > 0 and gvy > 0 else None
    glr = (gsx - gsy) / gN
    valid_samples = [s for s in per_sample if s["ess_frac"] is not None]
    mean_ess = (
        sum(s["ess_frac"] for s in valid_samples) / len(valid_samples)
        if valid_samples else None
    )
    return {
        "global": {
            "pearson": round(gpear, 5) if gpear is not None else None,
            "mean_log_ratio": round(glr, 5),
            "total_valid_tokens": int(gN),
            "mean_tokens_per_sample": round(gN / nrows, 1) if nrows else 0,
            "mean_ess_frac": round(mean_ess, 4) if mean_ess is not None else None,
            "hist": {"lo": lo, "hi": hi, "bins": HB, "counts": hist},
        },
        "per_sample": per_sample,
        "series": series,
        "series_points": n_series,
    }


# --------------------------------------------------------------------------
# main ingest
# --------------------------------------------------------------------------
def ingest(bundle_path: str, out_dir: str, progress=None) -> dict:
    """Materialize `bundle_path` into `out_dir`. Returns a summary dict
    (also written to out_dir/snapshot.json). `progress(done, total, msg)` is an
    optional callback."""

    def _p(done, total, msg):
        if progress:
            progress(done, total, msg)

    b = _Bundle(bundle_path)
    try:
        jname = b.find(suffix=".jsonl")
        tname = b.find(prefix="rollout_trajectories")
        oname = b.find(prefix="rollout_output")
        if not tname:
            raise ValueError("no rollout_trajectories_*.json in bundle")

        # 1) reward + output text from the jsonl (positional join)
        scores: list[float | None] = []
        jsonl_outputs: list[str] = []
        if jname:
            _p(0, 1, "reading jsonl rewards")
            with b.open(jname) as f:
                for line in f.read().decode("utf-8", "replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    o = json.loads(line)
                    scores.append(_fnum(o.get("score")))
                    jsonl_outputs.append(o.get("output") or "")

        trials_root = os.path.join(out_dir, "trials")
        os.makedirs(trials_root, exist_ok=True)

        # 2) stream trajectories, materialize trial dirs, collect stats.
        # The real async layout puts ONE prompt-group (its ~expected_n rollouts)
        # per step_NNNN dir; the dashboard's scanners rely on that (a step with
        # far more entries is treated as a validation dump and skipped). So we
        # give each instance its own step dir.
        from collections import Counter, defaultdict

        n = 0
        groups: dict[str, list[float | None]] = defaultdict(list)
        repos: dict[str, str] = {}
        resp_lens: list[int] = []
        logprob_means: list[float] = []
        row_rewards: list[float] = []  # positional reward per sample row
        term = Counter()
        reward_source = "jsonl_score" if scores else "is_correct"
        step_of: dict[str, int] = {}  # instance_id -> step number (1-based)

        _p(0, len(scores) or 256, "materializing trajectories")
        with b.open(tname) as fp:
            for i, t in enumerate(iter_trajectories(fp)):
                task = t.get("task", {})
                ea = task.get("extra_args", {})
                inst = ea.get("id") or task.get("task_id") or str(i)
                meta = ea.get("meta", {})
                repo = (meta.get("extra_info", {}) or {}).get("repo") or ""
                tr = (t.get("trajectories") or [{}])[0]
                chat = tr.get("chat_completions") or []

                # reward: positional jsonl score, else is_correct
                if i < len(scores) and scores[i] is not None:
                    reward = scores[i]
                else:
                    reward = 1.0 if str(t.get("is_correct")).lower() == "true" else 0.0

                rids = tr.get("response_ids") or []
                lps = tr.get("logprobs") or []
                resp_lens.append(len(rids))
                if lps:
                    fl = [_fnum(x) for x in lps]
                    fl = [x for x in fl if x is not None]
                    if fl:
                        logprob_means.append(sum(fl) / len(fl))
                term[str(t.get("termination_reason"))] += 1
                groups[inst].append(reward)
                row_rewards.append(reward)
                repos[inst] = repo

                if inst not in step_of:
                    step_of[inst] = len(step_of) + 1
                step_dir = os.path.join(trials_root, f"step_{step_of[inst]:04d}")
                # trial dir: <instance>-<idx6hex> so _strip_trial_hash() -> instance
                tdir = os.path.join(step_dir, f"{_safe(inst)}-{i:06x}")
                os.makedirs(os.path.join(tdir, "agent"), exist_ok=True)
                os.makedirs(os.path.join(tdir, "verifier"), exist_ok=True)
                traj = _chat_to_trajectory(chat)
                with open(os.path.join(tdir, "agent", "trajectory.json"), "w") as f:
                    json.dump(traj, f, ensure_ascii=False)
                with open(os.path.join(tdir, "verifier", "reward.txt"), "w") as f:
                    f.write(str(reward))
                n += 1
                if n % 16 == 0:
                    _p(n, len(scores) or n, f"materialized {n} trajectories")

        # 3) summary / metrics
        expected_n = max((len(v) for v in groups.values()), default=1)
        all_rewards = [r for v in groups.values() for r in v]
        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        solved = sum(1 for r in all_rewards if r >= 0.5)
        # correct-out-of-N histogram (advantage-zero groups at 0 and N)
        hist = [0] * (expected_n + 1)
        for v in groups.values():
            c = min(sum(1 for r in v if r >= 0.5), expected_n)
            hist[c] += 1
        zero_adv = hist[0] + hist[expected_n]
        rl = sorted(resp_lens)

        def pct(p):
            return rl[min(len(rl) - 1, int(len(rl) * p))] if rl else 0

        summary = {
            "format": "swerebench_openhands_sdk",
            "num_samples": n,
            "num_groups": len(groups),
            "expected_n": expected_n,
            "mean_reward": round(mean_reward, 4),
            "pass_rate": round(solved / n, 4) if n else 0.0,
            "solved": solved,
            "num_repos": len(set(repos.values())),
            "correct_out_of_n_hist": hist,
            "zero_advantage_groups": zero_adv,
            "resp_len": {
                "min": rl[0] if rl else 0,
                "p50": pct(0.5),
                "p90": pct(0.9),
                "max": rl[-1] if rl else 0,
                "mean": round(sum(rl) / len(rl), 1) if rl else 0,
            },
            "mean_logprob": round(sum(logprob_means) / len(logprob_means), 4)
            if logprob_means
            else None,
            "termination_reasons": dict(term),
            "reward_source": reward_source,
            "has_token_file": bool(oname),
            "token_file": os.path.basename(oname) if oname else None,
            "files": {
                "jsonl": os.path.basename(jname) if jname else None,
                "trajectories": os.path.basename(tname),
                "rollout_output": os.path.basename(oname) if oname else None,
            },
        }
        # 4) token-level IS / logprob diagnostics from the 1GB tensor file
        # (best-effort: a malformed token file must not fail the whole import).
        token_stats = None
        if oname:
            try:
                token_stats = _compute_token_stats(b, oname, row_rewards, _p)
            except Exception as e:
                token_stats = None
                summary["token_stats_error"] = str(e)
        if token_stats:
            g = token_stats["global"]
            summary["is_pearson"] = g["pearson"]
            summary["is_mean_log_ratio"] = g["mean_log_ratio"]
            summary["is_mean_ess_frac"] = g["mean_ess_frac"]
            summary["is_total_tokens"] = g["total_valid_tokens"]
            with open(os.path.join(out_dir, "token_stats.json"), "w") as f:
                json.dump(token_stats, f)
        summary["has_token_stats"] = bool(token_stats)

        # synthetic one-step metrics so Overview/Rewards/Metrics show a value
        metrics = [
            {
                "step": 1,
                "critic/score/mean": summary["mean_reward"],
                "critic/rewards/mean": summary["mean_reward"],
                "snapshot/pass_rate": summary["pass_rate"],
                "snapshot/zero_advantage_groups": zero_adv,
                "snapshot/num_groups": len(groups),
                "snapshot/resp_len_mean": summary["resp_len"]["mean"],
                "snapshot/mean_logprob": summary["mean_logprob"],
            }
        ]
        if token_stats:
            metrics[0]["snapshot/is_pearson"] = summary["is_pearson"]
            metrics[0]["snapshot/is_mean_log_ratio"] = summary["is_mean_log_ratio"]
            metrics[0]["snapshot/is_mean_ess_frac"] = summary["is_mean_ess_frac"]
        with open(os.path.join(out_dir, "snapshot.json"), "w") as f:
            json.dump({"summary": summary, "metrics": metrics}, f, indent=2)
        _p(n, n, "done")
        return summary
    finally:
        b.close()


if __name__ == "__main__":
    import sys

    src = sys.argv[1]
    out = sys.argv[2]
    s = ingest(src, out, progress=lambda d, t, m: print(f"  [{d}/{t}] {m}", flush=True))
    print(json.dumps(s, indent=2))
