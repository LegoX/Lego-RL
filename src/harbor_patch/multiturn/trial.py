"""Outer user-feedback loop; Harbor still owns setup, artifacts and cleanup.

The live /app and native CC session survive every round. Hidden verification
runs in a disposable environment made from a byte-preserving /app snapshot.
This adapter targets glm_ccbench_0629_v3.1, whose workspace is /app.
"""

import asyncio
import hashlib
import json
import math
import os
import shutil
import shlex
import time
from datetime import datetime, timezone
from uuid import uuid4

from harbor.environments.factory import EnvironmentFactory
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import TimingInfo
from harbor.trial.network_policy import (
    requires_mutable_network_after_start,
    resolve_agent_phase_policy,
    resolve_verifier_phase_policy,
    should_apply_agent_phase_policy,
    should_apply_environment_baseline,
    should_apply_legacy_baseline_for_dynamic_policy,
    should_apply_verifier_phase_policy,
)
from harbor.trial.trial import Trial, TrialEvent
from harbor.verifier.verifier import Verifier

from .benchmark import (
    BEHAVIORS,
    CATEGORY,
    PROFILE,
    MultiTurnBehaviorConfig,
    MultiTurnRoundResult,
)
from .config import MultiTurnConfig


class MultiTurnError(RuntimeError):
    """Invalid episode: never train against an earlier round's reward."""


async def create_multiturn_trial(config, options: MultiTurnConfig, proxy, session_id):
    if config.verifier.disable:
        raise ValueError("multi-turn RL requires the functional verifier")
    supported = {
        "harbor_patch.agents.image_mounted_claude_code:ClaudeCode",
        "harbor_patch.agents.image_mounted_claude_code.claude_code:ClaudeCode",
    }
    if config.agent.import_path not in supported:
        raise ValueError(
            "multi-turn currently supports the image-mounted Claude Code harness only"
        )
    config = config.model_copy(deep=True)
    config.agent.import_path = "harbor_patch.multiturn.claude_code:MultiTurnClaudeCode"
    task = await MultiTurnTrial._load_task(config)
    # Kubernetes may infer docker_image by mutating TaskConfig in its constructor.
    # Keep the pristine definition so each verifier still performs Dockerfile
    # setup/COPY, rather than silently running only the base image.
    verifier_env_config = task.config.environment.model_copy(deep=True)
    if config.environment.import_path in {
        "harbor_patch.environments.kubernetes.kubernetes:KubernetesEnvironment",
        "harbor_patch.multiturn.kubernetes:MultiTurnKubernetesEnvironment",
    }:
        has_prebuilt_openswe = (options.task_profile == "openswe"
            and bool(os.environ.get("HARBOR_OPENSWE_IMAGE_REGISTRY"))
            and "openswe" in str(task.paths.environment_dir).lower())
        if not verifier_env_config.docker_image and not has_prebuilt_openswe and os.environ.get(
            "HARBOR_K8S_INLINE_BUILD", ""
        ).lower() not in {"1", "true", "yes", "on"}:
            raise ValueError(
                "Benchmark tasks without prebuilt images require HARBOR_K8S_INLINE_BUILD=true"
            )
        config.environment.import_path = (
            "harbor_patch.multiturn.kubernetes:MultiTurnKubernetesEnvironment"
        )
        config.environment.kwargs["workspace_dir"] = options.resolved_workspace
    trial = MultiTurnTrial(config, _task=task)
    trial._verifier_env_config = verifier_env_config
    trial.options = options
    trial.on_round_start = lambda round_id: proxy.begin_user_round(session_id, round_id)
    return trial


class MultiTurnTrial(Trial):
    async def _execute_agent(self):
        started = time.monotonic()
        self._rounds = []
        self._final_verdict = None
        self._final_verifier_paths = None
        self._stop_reason = "error"
        self._simulator = None
        self._judge = None
        self._phase = "agent"
        self.result.agent_result = AgentContext(metadata={})
        self.result.agent_execution = TimingInfo(started_at=datetime.now(timezone.utc))
        await self._invoke_hooks(TrialEvent.AGENT_START)
        try:
            if should_apply_agent_phase_policy(self._task.config, self.config.agent):
                await self._environment.apply_network_policy(
                    resolve_agent_phase_policy(
                        self._task.config,
                        self.config.agent,
                        self._task.config.environment.resolve_baseline(),
                    )
                )
            async with asyncio.timeout(self.options.episode_timeout_sec):
                await self._run_user_rounds()
        except asyncio.CancelledError:
            self._final_verdict = None
            self.result.verifier_result = None
            self._stop_reason = "cancelled"
            raise
        except Exception as exc:
            self._final_verdict = None
            self.result.verifier_result = None
            # Keep the original details in exception chaining / exception.txt,
            # not in the legacy loop's string-based nonzero/overflow classifier.
            raise MultiTurnError(
                "Multi-turn episode failed; see exception.txt"
            ) from exc
        finally:
            self.result.agent_execution.finished_at = datetime.now(timezone.utc)
            metadata = {
                "enabled": True,
                "schema_version": 1,
                "task_profile": self.options.task_profile,
                "test_metric": "task_success" if self.options.task_profile == "openswe" else "test_pass_rate",
                "valid": self._final_verdict is not None,
                "duration_sec": time.monotonic() - started,
                "failure_stage": self._phase if self._final_verdict is None else None,
                "num_user_rounds": len(self._rounds),
                "stop_reason": self._stop_reason,
                "simulator_model": self.options.simulator_model,
                "simulator_mode": self.options.simulator_mode,
                "max_rounds": self.options.max_rounds,
                "seed": self.options.seed,
                "reward_mode": self.options.reward_mode,
                "judge_model": self.options.judge_model,
                "round_rewards": [{k: r[k] for k in (
                    "round_id", "functional_reward", "judge_score", "training_reward"
                ) if k in r} for r in self._rounds if r.get("status") == "completed"],
                "workspace_snapshot": self.options.resolved_workspace + " (files only; excludes running processes and system installs)",
            }
            self.result.agent_result.metadata["multiturn"] = metadata
            (self._trial_paths.trial_dir / "multiturn.json").write_text(
                json.dumps(
                    {**metadata, "rounds": self._rounds}, ensure_ascii=False, indent=2
                )
                + "\n",
                encoding="utf-8",
            )

    async def _run_user_rounds(self):
        instruction = self._task.instruction
        for round_id in range(1, self.options.max_rounds + 1):
            round_dir = (
                self._trial_paths.trial_dir / "multiturn" / f"round-{round_id:03d}"
            )
            round_dir.mkdir(parents=True, exist_ok=True)
            (round_dir / "instruction.txt").write_text(instruction, encoding="utf-8")
            await self.on_round_start(round_id)
            record = {
                "round_id": round_id,
                "instruction": instruction,
                "status": "running",
            }
            self._rounds.append(record)
            self._phase = "agent"
            agent_result = await asyncio.wait_for(
                self._agent.run_user_round(
                    instruction,
                    self._environment,
                    self.result.agent_result,
                    round_id,
                    round_dir / "claude-code.jsonl",
                ),
                timeout=self._agent_timeout_sec,
            )
            record.update(agent_result)
            self._phase = "verifier"
            verdict, paths = await self._verify_round(round_id, round_dir)
            reward = verdict.rewards.get("reward") if verdict.rewards else None
            if (
                not isinstance(reward, (int, float))
                or not math.isfinite(reward)
                or not 0 <= reward <= 1
            ):
                raise MultiTurnError(
                    "Benchmark verifier must emit a finite reward in [0, 1]"
                )
            record.update(functional_reward=reward)
            training_reward = float(reward)
            if self.options.judge_enabled:
                self._phase = "judge"
                from .round_judge import RoundJudge, build_round_context

                if self._judge is None:
                    self._judge = RoundJudge(self.options)
                context = build_round_context(
                    self._task, self._rounds, self._trial_paths.trial_dir,
                    self.options.judge_max_context_chars,
                )
                context["task_profile"] = self.options.task_profile
                judgment = await self._judge.score(context, round_dir)
                # The benchmark's per-round metric uses correctness, not its
                # separately reported whole-UX reward.
                judge_score = judgment.task_success_score / 100.0
                record.update(judge_score=judge_score, judge_result=judgment.model_dump())
                training_reward = (self.options.test_weight * reward
                                   + self.options.judge_weight * judge_score)
            record.update(status="completed", training_reward=training_reward)
            self._final_verdict = verdict.model_copy(deep=True)
            self._final_verdict.rewards = {"reward": training_reward}
            self._final_verifier_paths = paths
            if agent_result.get("stop_reason") == "context_limit":
                # Verify and judge the current workspace before committing this
                # truncated round. Never substitute an earlier round's reward.
                self._stop_reason = "context_limit"
                return
            if self.options.stop_on_success and reward == 1:
                self._stop_reason = "verifier_success"
                return
            if round_id == self.options.max_rounds:
                self._stop_reason = "max_rounds"
                return
            self._phase = "simulator"
            instruction = await self._next_instruction(round_id + 1, round_dir)
            if instruction is None:
                self._stop_reason = "simulator_no_op"
                return

    async def _next_instruction(self, round_id, round_dir):
        if self.options.simulator_mode == "scripted":
            index = round_id - 2
            if index >= len(self.options.scripted_followups):
                return None
            return self.options.scripted_followups[index].strip() or None
        from .user_simulator import UserSimulator, UserSimulatorInput

        if self._simulator is None:
            self._simulator = UserSimulator(
                model=self.options.simulator_model,
                api_base=self.options.simulator_api_base,
                api_key=os.environ.get(self.options.simulator_api_key_env),
                temperature=self.options.simulator_temperature,
            )
        # Stable selection across Ray workers/restarts; no Python hash randomization.
        digest = hashlib.sha256(
            f"{self.options.seed}:{self._task.name}:{round_id}".encode()
        ).digest()
        behavior_id = CATEGORY["behavior_ids"][
            int.from_bytes(digest[:4], "big") % len(CATEGORY["behavior_ids"])
        ]
        previous = self._rounds[-1]
        # Hidden tests, raw verifier logs, exact rewards and API credentials never
        # enter the simulator prompt. Native agent logs stay available for audit.
        behavior = dict(BEHAVIORS[behavior_id])
        profile = dict(PROFILE)
        if self.options.task_profile == "openswe":
            behavior["instruction"] += (
                " Preserve the original issue's acceptance criteria. Do not invent new requirements, "
                "test failures, error messages or reproduction evidence. Ask for diagnosis when evidence is absent."
            )
            profile["task_stratum"] = "openswe_issue_repair"
        payload = UserSimulatorInput(
            task_name=self._task.name,
            initial_instruction=self._task.instruction,
            round_id=round_id,
            behavior=MultiTurnBehaviorConfig(id=behavior_id, **behavior),
            previous_rounds=[
                MultiTurnRoundResult(
                    round_id=r["round_id"], instruction=r["instruction"]
                )
                for r in self._rounds
            ],
            context_summary={
                "final_message": str(previous.get("final_message", "")),
                "round_verifier_feedback": {
                    "functional_reward": previous["functional_reward"],
                    "history": [{"round_id": r["round_id"], "functional_reward": r["functional_reward"]}
                                for r in self._rounds if "functional_reward" in r],
                    "outcome": "passed"
                    if previous["functional_reward"] == 1
                    else "not_passed",
                },
            },
            seed=self.options.seed,
            profile_details=profile,
            task_category="higher-test-case-pass-rate",
            task_category_details=CATEGORY,
        )
        for attempt in range(self.options.simulator_max_retries):
            try:
                output = await asyncio.wait_for(
                    self._simulator.generate(
                        payload, round_dir / f"simulator-{attempt}.json"
                    ),
                    timeout=self.options.simulator_timeout_sec,
                )
                (round_dir / "simulated-user.json").write_text(
                    output.model_dump_json(indent=2), encoding="utf-8"
                )
                return None if output.action == "no-op" else output.instruction
            except Exception:
                if attempt + 1 == self.options.simulator_max_retries:
                    raise
        raise AssertionError("unreachable")

    async def _verify_round(self, round_id, round_dir):
        """Run the unmodified benchmark verifier away from the policy sandbox."""
        await self._invoke_hooks(TrialEvent.VERIFICATION_START)
        if self.result.verifier is None:
            self.result.verifier = TimingInfo(started_at=datetime.now(timezone.utc))
        paths = TrialPaths(trial_dir=round_dir / "verification")
        paths.mkdir()
        env = EnvironmentFactory.create_environment_from_config(
            config=self.config.environment.model_copy(deep=True),
            environment_dir=self._task.paths.environment_dir,
            environment_name=self._task.name,
            session_id=f"mt-verify-{uuid4().hex[:16]}",
            trial_paths=paths,
            task_env_config=self._verifier_env_config.model_copy(deep=True),
            logger=self._logger,
            **(
                {
                    "has_dynamic_network_policy": requires_mutable_network_after_start(
                        self._task.config, self.config.agent
                    )
                }
                if self.config.environment.type == "docker"
                else {}
            ),
        )
        workspace = shlex.quote(self.options.resolved_workspace)
        archive = round_dir / "workspace.tar"
        remote_archive = f"/tmp/mt-workspace-{uuid4().hex}.tar"
        try:
            await self._checked_exec(
                self._environment,
                f"test -d {workspace} && test ! -L {workspace} && tar -C {workspace} -cf {remote_archive} .",
            )
            await self._environment.download_file(remote_archive, archive)
            await asyncio.wait_for(
                env.start(force_build=self.config.environment.force_build),
                self._environment_build_timeout_sec,
            )
            await env.run_healthcheck()
            if should_apply_environment_baseline(self._task.config) or (
                env.supports_dynamic_network_policy
                and should_apply_legacy_baseline_for_dynamic_policy(
                    self._task.config, self.config.agent
                )
            ):
                await env.apply_network_policy(
                    self._task.config.environment.resolve_baseline()
                )
            if should_apply_verifier_phase_policy(self._task.config, self.config.agent):
                await env.apply_network_policy(
                    resolve_verifier_phase_policy(
                        self._task.config,
                        self._task.config.environment.resolve_baseline(),
                    )
                )
            await env.upload_file(archive, remote_archive)
            await self._checked_exec(
                env,
                (
                    f"test -d {workspace} && test ! -L {workspace} && "
                    f"find {workspace} -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} + && "
                    f"tar -C {workspace} -xf {remote_archive}"
                ),
            )
            verdict = await asyncio.wait_for(
                Verifier(
                    task=self._task,
                    trial_paths=paths,
                    environment=env,
                    override_env=self.config.verifier.env or None,
                ).verify(),
                timeout=self._verifier_timeout_sec,
            )
            # Some benchmark browser checks swallow infrastructure errors and
            # still emit reward=1. Such a run is not evidence of task success.
            stdout = (
                paths.test_stdout_path.read_text(errors="replace")
                if paths.test_stdout_path.exists()
                else ""
            )
            if "treated as non-blocking" in stdout:
                raise MultiTurnError("Benchmark verifier skipped a browser check")
            return verdict, paths
        finally:
            self.result.verifier.finished_at = datetime.now(timezone.utc)
            archive.unlink(missing_ok=True)
            try:
                await self._environment.exec(f"rm -f {remote_archive}", user="root")
            finally:
                await asyncio.shield(env.stop(delete=True))

    @staticmethod
    async def _checked_exec(environment, command):
        result = await environment.exec(command=command, user="root")
        if result.return_code != 0:
            raise MultiTurnError("Workspace snapshot operation failed")

    async def _run_verification(self):
        # super().run() reaches this hook once after the whole episode. Reuse
        # the final isolated verdict; never run hidden tests in the live /app.
        if self._final_verdict is None:
            raise MultiTurnError("No valid final multi-turn verdict")
        self.result.verifier_result = self._final_verdict
        shutil.copytree(
            self._final_verifier_paths.verifier_dir,
            self._trial_paths.verifier_dir,
            dirs_exist_ok=True,
        )
