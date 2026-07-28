"""OpenHands SDK agent adapter for Harbor.

This adapter allows running the OpenHands Software Agent SDK inside
Harbor-managed containers for benchmarking and evaluation.
"""

import json
import os
import re
import shlex
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.agents.utils import get_api_key_var_names_from_model_name
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName
from harbor.models.trial.paths import EnvironmentPaths

from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    TemplateSyntaxError,
    UndefinedError,
    meta,
)


class OpenHandsSDK(BaseInstalledAgent):
    """
    The OpenHands SDK agent uses the OpenHands Software Agent SDK to solve tasks.

    Unlike the full OpenHands (openhands-ai) which includes a Docker runtime,
    this adapter uses the lightweight SDK that runs directly in the container.
    """

    SUPPORTS_ATIF: bool = True
    _OUTPUT_FILENAME = "openhands_sdk.txt"
    _TRAJECTORY_FILENAME = "trajectory.json"

    DEFAULT_SKILL_PATHS = [
        "~/.openhands-sdk/skills",
        "~/.claude/skills",
        "~/.codex/skills",
        "~/.agents/skills",
        "~/.goose/skills",
        "~/.gemini/skills",
        "~/.factory/skills",
        "~/.opencode/skill",
    ]

    def __init__(
        self,
        reasoning_effort: str | None = "high",
        load_skills: bool = True,
        skill_paths: list[str] | None = None,
        collect_token_ids: bool = False,
        max_iterations: int | None = None,
        temperature: float | None = None,
        api_base: str | None = None,
        model_info: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ):
        """
        Initialize OpenHands SDK agent.

        Args:
            reasoning_effort: Reasoning effort level (low, medium, high).
            load_skills: Whether to load skills from skill paths.
            skill_paths: Custom skill paths to load from. If None, uses default paths.
            collect_token_ids: When True, request token IDs from the LLM backend
                (requires SGLang/vLLM; third-party APIs will ignore this).
            max_iterations: Maximum number of agent iterations per run.
                Maps to the SDK's max_iteration_per_run parameter.
            temperature: LLM sampling temperature (0.0 to 2.0).
            api_base: Custom API base URL for the LLM backend.
            model_info: Model configuration dictionary with token limits.
                Supports "max_input_tokens" and "max_output_tokens" keys.
        """
        super().__init__(*args, **kwargs)
        self._reasoning_effort = reasoning_effort
        self._load_skills = load_skills
        self._skill_paths = skill_paths or self.DEFAULT_SKILL_PATHS
        self._collect_token_ids = collect_token_ids
        self._max_iterations = max_iterations
        self._temperature = temperature
        self._api_base = api_base
        self._model_info = model_info

    @staticmethod
    def name() -> str:
        return AgentName.OPENHANDS_SDK.value

    def get_version_command(self) -> str | None:
        return "/opt/openhands-sdk-venv/bin/pip show openhands-sdk | grep ^Version:"
        # return "/installed-agent/openhands-sdk-venv/bin/pip show openhands-sdk | grep ^Version:"

    def parse_version(self, stdout: str) -> str:
        # Output: "Version: 0.1.2"
        text = stdout.strip()
        if text.startswith("Version:"):
            return text.removeprefix("Version:").strip()
        return text

    @property
    def _trajectory_path(self) -> PurePosixPath:
        return PurePosixPath(EnvironmentPaths.agent_dir / self._TRAJECTORY_FILENAME)

    async def install(self, environment: BaseEnvironment) -> None:
        # Check if already installed
        check_result = await environment.exec(
            command='[ -f /opt/openhands-sdk-venv/bin/python ] && /opt/openhands-sdk-venv/bin/python -c "import openhands.sdk" 2>/dev/null',
        )
        already_installed = check_result.return_code == 0

        if not already_installed:
            # Install python3-venv if needed (root)
            await self.exec_as_root(
                environment,
                command=(
                    "mkdir -p /opt && "
                    'if ! command -v python3.12 >/dev/null 2>&1; then'
                    "  apt-get update -qq && "
                    "  apt-get install -y software-properties-common && "
                    "  add-apt-repository -y ppa:deadsnakes/ppa && "
                    "  apt-get update -qq && "
                    "  apt-get install -y python3.12 python3.12-venv python3.12-dev;"
                    "fi"
                ),
                env={"DEBIAN_FRONTEND": "noninteractive"},
            )
            # Create venv dir owned by default user
            agent_user = environment.default_user or "root"
            await self.exec_as_root(
                environment,
                command=f"mkdir -p /opt/openhands-sdk-venv && chown {agent_user}:{agent_user} /opt/openhands-sdk-venv",
            )
            # Install SDK (as default user)
            version_spec = f"=={self._version}" if self._version else ""
            await self.exec_as_agent(
                environment,
                command=(
                    "set -euo pipefail; "
                    'python3.12 -c "import sys; assert sys.version_info >= (3, 12)" && '
                    "python3.12 -m venv /opt/openhands-sdk-venv && "
                    "source /opt/openhands-sdk-venv/bin/activate && "
                    "export PIP_DEFAULT_TIMEOUT=120 && "
                    "pip install --upgrade pip && "
                    f"pip install openhands-sdk{version_spec} openhands-tools{version_spec}"
                ),
            )

        # Upload runner script
        runner_script_path = Path(__file__).parent / "openhands_sdk_runner.py"
        local_copy = self.logs_dir / "run_agent.py"
        local_copy.write_text(runner_script_path.read_text())
        await environment.upload_file(
            source_path=local_copy,
            target_path="/installed-agent/run_agent.py",
        )
        await environment.exec(
            command="chmod +x /installed-agent/run_agent.py",
            user="root",
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        """
        Populate context with results from agent trajectory.
        """
        trajectory_file = self.logs_dir / self._TRAJECTORY_FILENAME
        if not trajectory_file.exists():
            self.logger.warning(f"No trajectory file found at {trajectory_file}")
            return

        try:
            with open(trajectory_file) as f:
                trajectory_data = json.load(f)

            # Extract metrics from trajectory
            final_metrics = trajectory_data.get("final_metrics", {})
            context.cost_usd = final_metrics.get("total_cost_usd")
            context.n_input_tokens = final_metrics.get("total_prompt_tokens", 0)
            context.n_output_tokens = final_metrics.get("total_completion_tokens", 0)
            context.n_cache_tokens = final_metrics.get("total_cached_tokens", 0)
            if context.metadata is None:
                context.metadata = {}
            message_trajectory_file = trajectory_file.with_suffix(".message.json")
            with open(message_trajectory_file) as f:
                message_trajectory_data = json.load(f)
            context.metadata["all_messages"] = message_trajectory_data["messages"]
            context.metadata["tools"] = message_trajectory_data["tools"]

        except (json.JSONDecodeError, OSError) as e:
            self.logger.error(f"Failed to parse trajectory file: {e}")

    def render_prompt_template(self, template_path: Path, raw_instruction: str) -> str:
        """
        Parses the raw instruction block, extracts metadata, and renders the template.

        Args:
            template_path: Path to the Jinja2 template.
            raw_instruction: The raw string containing # Task, main body, and Repo/Commit info.
        """
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

        # --- 1. Extract information using regex ---
        # Extract the last part of the Repo name (e.g., astropy/astropy -> astropy)
        repo_match = re.search(r"\*\*Repo:\*\*\s+`?.*?/([^`\s]+)`?", raw_instruction)
        testbed_dir_name = repo_match.group(1).strip() if repo_match else ""

        # Extract Base commit
        commit_match = re.search(r"\*\*Base commit:\*\*\s+`?([a-f0-9]+)`?", raw_instruction)
        base_commit = commit_match.group(1).strip() if commit_match else ""

        # --- 2. Clean instruction text ---
        # a. Remove the leading "# Task" (case-insensitive)
        clean_instruction = re.sub(r"^#\s*Task\s*", "", raw_instruction, flags=re.IGNORECASE).strip()
        
        # b. Remove the trailing metadata block (strip everything from "---\n\n**Repo:**" onwards)
        # We look for the first occurrence of the "---\n\n**Repo:**" marker
        metadata_start = re.search(r"---(?:\s*\n)+\*\*Repo:\*\*", clean_instruction)
        if metadata_start:
            clean_instruction = clean_instruction[:metadata_start.start()].strip()

        # --- 3. Jinja2 rendering logic ---
        template_content = template_path.read_text()
        
        # Static check (optional, ensures the template contains required variables)
        env_for_parsing = Environment()
        try:
            ast = env_for_parsing.parse(template_content)
            vars_in_template = meta.find_undeclared_variables(ast)
            if "instruction" not in vars_in_template:
                raise ValueError(f"Template {template_path} is missing '{{{{ instruction }}}}'")
        except TemplateSyntaxError as e:
            raise ValueError(f"Template syntax error: {e}")

        try:
            env = Environment(undefined=StrictUndefined)
            template = env.from_string(template_content)
            
            # Render using the extracted variable names
            return template.render(
                instruction=clean_instruction,
                testbed_dir_name=testbed_dir_name,
                base_commit=base_commit
            )
        except UndefinedError as e:
            raise ValueError(f"Missing variable in template: {e}")
        except Exception as e:
            raise ValueError(f"Rendering failed: {e}")
    
    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        """Run the OpenHands SDK agent."""
        current_dir = Path(__file__).parent
        template_path = current_dir / "user_template.j2"
        instruction = self.render_prompt_template(template_path, instruction)

        escaped_instruction = shlex.quote(instruction)

        env: dict[str, str] = {}

        is_hosted_vllm_model = (
            self.model_name is not None
            and self.model_name.lower().startswith("hosted_vllm/")
        )

        # Handle LLM API key with fallback logic
        # For hosted_vllm models (local vLLM servers), use a dummy key if none set
        if "LLM_API_KEY" in os.environ:
            env["LLM_API_KEY"] = os.environ["LLM_API_KEY"]
        elif is_hosted_vllm_model:
            # Local vLLM servers don't validate API keys, use dummy
            env["LLM_API_KEY"] = "dummy-key-for-local-vllm"
        else:
            if self.model_name:
                try:
                    api_key_vars = get_api_key_var_names_from_model_name(
                        self.model_name
                    )
                    if len(api_key_vars) == 1:
                        api_key_var = api_key_vars[0]
                        if api_key_var in os.environ:
                            env["LLM_API_KEY"] = os.environ[api_key_var]
                        else:
                            raise ValueError(
                                f"Unset API variable found for model {self.model_name}. "
                                f"Please set {api_key_var} or LLM_API_KEY environment variable"
                            )
                    else:
                        for api_key_var in api_key_vars:
                            if api_key_var in os.environ:
                                # Use model-agnostic variables received by OpenHands
                                oh_api_key_var = api_key_var.replace("AZURE_", "LLM_")
                                env[oh_api_key_var] = os.environ[api_key_var]
                            else:
                                raise ValueError(
                                    f"Unset API variable found for model {self.model_name}. "
                                    f"Please set {api_key_var} or LLM_API_KEY environment variable"
                                )
                except ValueError as e:
                    raise ValueError(
                        f"Unable to determine API key for model {self.model_name}: {e}. "
                        "Please set LLM_API_KEY environment variable as fallback"
                    )
            else:
                raise ValueError(
                    "No LLM API key found and no model specified. "
                    "Please set LLM_API_KEY environment variable or specify a model"
                )

        # Set API base URL with prioritization: api_base param > env var > standard defaults
        if self._api_base:
            env["LLM_BASE_URL"] = self._api_base
        elif "LLM_BASE_URL" in os.environ:
            env["LLM_BASE_URL"] = os.environ["LLM_BASE_URL"]
        else:
            # Check for common API base environment variable names
            for candidate in ("HOSTED_VLLM_API_BASE", "VLLM_API_BASE", "OPENAI_API_BASE"):
                if candidate in os.environ:
                    env["LLM_BASE_URL"] = os.environ[candidate]
                    break

        # Set model name with prioritization: model_name param > env var
        if self.model_name:
            env["LLM_MODEL"] = self.model_name
        elif "LLM_MODEL" in os.environ:
            env["LLM_MODEL"] = os.environ["LLM_MODEL"]
        elif "ANTHROPIC_MODEL" in os.environ:
            env["LLM_MODEL"] = os.environ["ANTHROPIC_MODEL"]
        else:
            raise ValueError("No LLM model specified")

        # Set model info token limits (critical for hosted_vllm models)
        # Without these, the SDK may use default limits that are too restrictive
        if self._model_info:
            if self._model_info.get("max_input_tokens"):
                env["LLM_MAX_INPUT_TOKENS"] = str(self._model_info["max_input_tokens"])
            if self._model_info.get("max_output_tokens"):
                env["LLM_MAX_OUTPUT_TOKENS"] = str(self._model_info["max_output_tokens"])
        if "LLM_NATIVE_TOOL_CALLING" in os.environ:
            env["LLM_NATIVE_TOOL_CALLING"] = os.environ["LLM_NATIVE_TOOL_CALLING"]
        # env["LLM_NATIVE_TOOL_CALLING"] = "False"

        # Set up paths
        env["AGENT_LOGS_DIR"] = "/logs/agent"
        env["TRAJECTORY_PATH"] = f"/logs/agent/{self._TRAJECTORY_FILENAME}"
        env["LOAD_SKILLS"] = "1" if self._load_skills else "0"
        env["SKILL_PATHS"] = ":".join(self._skill_paths)

        # Pass MCP server config so run_agent.py can register them with the SDK
        if self.mcp_servers:
            mcp_list: list[dict[str, str | list[str]]] = []
            for server in self.mcp_servers:
                entry: dict[str, str | list[str]] = {
                    "name": server.name,
                    "transport": server.transport,
                }
                if server.transport == "stdio":
                    if server.command:
                        entry["command"] = server.command
                    if server.args:
                        entry["args"] = server.args
                else:
                    if server.url:
                        entry["url"] = server.url
                mcp_list.append(entry)
            env["MCP_SERVERS_JSON"] = json.dumps(mcp_list)

        # Pass litellm extra_body for token ID collection (e.g. vLLM backends)
        if self._collect_token_ids:
            env["LITELLM_EXTRA_BODY"] = json.dumps({"return_token_ids": True})

        if self._max_iterations is not None:
            env["MAX_ITERATIONS"] = str(self._max_iterations)

        if self._temperature is not None:
            env["LLM_TEMPERATURE"] = str(self._temperature)

        # Build the command that runs our agent script
        command = f"""
/opt/openhands-sdk-venv/bin/python /installed-agent/run_agent.py \
    --instruction={escaped_instruction} \
    --logs-dir="$AGENT_LOGS_DIR" \
    --trajectory-path="$TRAJECTORY_PATH" \
    2>&1 | stdbuf -oL tee /logs/agent/{self._OUTPUT_FILENAME}
"""

        await self.exec_as_agent(environment, command=command.strip(), env=env)
