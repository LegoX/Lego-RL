#!/usr/bin/env python3
"""Harbor runner script for OpenHands SDK agent."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from openhands.sdk import (
    LLM,
    Agent,
    AgentContext,
    Conversation,
    Tool,
    get_logger,
)
from openhands.sdk.context import Skill
from openhands.sdk.event import (
    ActionEvent,
    MessageEvent,
    ObservationEvent,
    TokenEvent,
)
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

logger = get_logger(__name__)


def load_skill_from_file(skill_path: Path) -> Skill | None:
    """Load a skill from a SKILL.md file."""
    if not skill_path.exists():
        return None

    content = skill_path.read_text()
    name = skill_path.parent.name

    return Skill(
        name=name,
        content=content,
        source=str(skill_path),
        trigger=None,  # Always active
    )


def discover_skills(skill_paths: list[str]) -> list[Skill]:
    """Discover skills from SkillsBench skill paths."""
    seen_names: set[str] = set()
    skills: list[Skill] = []

    for base_path_str in skill_paths:
        base_path = Path(base_path_str).expanduser()
        if not base_path.exists():
            continue

        # Look for SKILL.md files in immediate subdirectories
        for skill_dir in base_path.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                skill = load_skill_from_file(skill_file)
                if skill and skill.name not in seen_names:
                    seen_names.add(skill.name)
                    skills.append(skill)
                    logger.info(f"Loaded skill: {skill.name} from {skill_file}")

    return skills


def build_trajectory(
    events: list[dict[str, Any]],
    llm_metrics: dict[str, Any],
    model_name: str,
    system_prompt: str | None = None,
    tool_definitions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an ATIF-format trajectory from conversation events."""
    steps: list[dict[str, Any]] = []
    step_id = 1

    for event in events:
        event_type = event.get("type", "")

        if event_type == "user_message":
            steps.append(
                {
                    "step_id": step_id,
                    "timestamp": event.get("timestamp"),
                    "source": "user",
                    "message": event.get("content", ""),
                }
            )
            step_id += 1

        elif event_type == "assistant_message":
            step: dict[str, Any] = {
                "step_id": step_id,
                "timestamp": event.get("timestamp"),
                "source": "agent",
                "message": event.get("content", ""),
                "model_name": model_name,
            }

            # Add tool calls if present
            tool_calls = event.get("tool_calls", [])
            if tool_calls:
                step["tool_calls"] = [
                    {
                        "tool_call_id": tc.get("id", ""),
                        "function_name": tc.get("name", ""),
                        "arguments": tc.get("arguments", {}),
                    }
                    for tc in tool_calls
                ]

            token_data = event.get("token_ids")
            if token_data:
                step["metrics"] = {
                    "prompt_token_ids": token_data.get("prompt_token_ids", []),
                    "completion_token_ids": token_data.get("response_token_ids", []),
                }

            steps.append(step)
            step_id += 1

        elif event_type == "tool_result":
            # Find the previous step and add observation
            if steps and steps[-1].get("source") == "agent":
                steps[-1]["observation"] = {
                    "results": [
                        {
                            "source_call_id": event.get("tool_call_id"),
                            "content": event.get("content", ""),
                        }
                    ]
                }

    if system_prompt:
        system_step: dict[str, Any] = {
            "step_id": 0,
            "timestamp": steps[0]["timestamp"] if steps else None,
            "source": "system",
            "message": system_prompt,
        }
        steps.insert(0, system_step)

    for i, step in enumerate(steps):
        step["step_id"] = i + 1

    trajectory = {
        "schema_version": "ATIF-v1.5",
        "session_id": os.environ.get("SESSION_ID", "harbor-session"),
        "agent": {
            "name": "openhands-sdk",
            "tool_definitions": tool_definitions if tool_definitions else None,
            "version": "unknown",  # Will be filled by SDK
        },
        "steps": steps,
        "final_metrics": {
            "total_prompt_tokens": llm_metrics.get("prompt_tokens", 0),
            "total_completion_tokens": llm_metrics.get("completion_tokens", 0),
            "total_cached_tokens": llm_metrics.get("cached_tokens", 0),
            "total_cost_usd": llm_metrics.get("cost_usd", 0.0),
        },
    }

    return trajectory


def dump_conversation_events(conversation, out_path: str | Path = "events_dump.json"):
    """
    Dump conversation.state.events -> JSON file for debugging.

    Args:
        conversation: a Conversation instance (exposes conversation.state.events)
        out_path: where to write the dump (default ./events_dump.json)
    """
    events = list(conversation.state.events)  # copy, so concurrent mutation cannot bite
    serialized = []

    for ev in events:
        try:
            # Prefer Pydantic's JSON-safe export.
            ev_dict = ev.model_dump(mode="json", exclude_none=True)
        except Exception:
            try:
                # Fall back to a plain model_dump.
                ev_dict = ev.model_dump(exclude_none=True)
            except Exception:
                # Last resort: a string representation, so this never raises.
                ev_dict = {"__repr__": repr(ev), "type": type(ev).__name__}
        serialized.append(ev_dict)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(serialized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Dumped {len(serialized)} events to {out_path.resolve()}")


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert an object into a JSON-friendly structure.
    - prefers pydantic's model_dump(mode='json') when available
    - handles dict / list / primitive types
    - anything else falls back to str()
    """
    # Primitives are already serializable.
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Already a dict / list / tuple -> recurse.
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_serializable(v) for v in obj]

    # Pydantic model_dump takes precedence.
    try:
        if hasattr(obj, "model_dump"):
            return _make_json_serializable(obj.model_dump(mode="json", exclude_none=True))
    except Exception:
        pass

    # A dataclass, or anything exposing dict().
    try:
        if hasattr(obj, "dict"):
            return _make_json_serializable(obj.dict())
    except Exception:
        pass

    # Last resort: the string representation.
    try:
        return str(obj)
    except Exception:
        return repr(obj)

from openhands.sdk.event import LLMConvertibleEvent
def dump_openai_chat_messages(conversation, llm, tool_definitions=None, out_path: str | Path = "openai_messages.json"):
    """
    Build OpenAI-style messages from conversation.state.events and write them to JSON.

    Args:
        conversation: an openhands Conversation (finished, or read in a safe state)
        llm: an openhands.sdk.LLM instance (used for format_messages_for_llm)
        out_path: where to write the messages
    """
    # 1) Keep only the events the LLM can see.
    llm_events = [e for e in conversation.state.events if isinstance(e, LLMConvertibleEvent)]

    # 2) Events -> the SDK's intermediate Message form (this merges multi-action steps).
    llm_messages = LLMConvertibleEvent.events_to_messages(llm_events)

    # 3) The LLM layer formats them as OpenAI Chat-style messages (a list of dicts).
    openai_chat_messages = llm.format_messages_for_llm(llm_messages)

    # 4) Make it serializable and write it out.
    serializable = {"messages": _make_json_serializable(openai_chat_messages),
                    "tools": _make_json_serializable(tool_definitions)}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(serializable) if isinstance(serializable, list) else 1} messages to {out_path.resolve()}")
    return serializable

def main():
    parser = argparse.ArgumentParser(description="Run OpenHands SDK agent")
    parser.add_argument("--instruction", required=True, help="Task instruction")
    parser.add_argument("--logs-dir", required=True, help="Directory for logs")
    parser.add_argument(
        "--trajectory-path", required=True, help="Path to save trajectory"
    )
    args = parser.parse_args()

    # Get configuration from environment
    model = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4-5-20250929")
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")

    if not api_key:
        print("Error: LLM_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    # Create logs directory
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Parse optional litellm extra body (for token ID collection with SGLang/vLLM)
    litellm_extra_body: dict[str, Any] = {}
    extra_body_raw = os.environ.get("LITELLM_EXTRA_BODY")
    if extra_body_raw:
        litellm_extra_body = json.loads(extra_body_raw)
        logger.info(f"LiteLLM extra body: {litellm_extra_body}")

    # Configure LLM
    llm_kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
    }
    if litellm_extra_body:
        llm_kwargs["litellm_extra_body"] = litellm_extra_body
    temperature_raw = os.environ.get("LLM_TEMPERATURE")
    if temperature_raw:
        llm_kwargs["temperature"] = float(temperature_raw)
    # native_tool_calling_raw = os.environ.get("LLM_NATIVE_TOOL_CALLING")
    # if native_tool_calling_raw:
    #     llm_kwargs["native_tool_calling"] = native_tool_calling_raw in ("true", "1", "True")
    llm = LLM(**llm_kwargs)

    # Configure tools
    tools = [
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
    ]

    # Load skills if enabled
    skills: list[Skill] = []
    if os.environ.get("LOAD_SKILLS", "1") == "1":
        skill_paths_str = os.environ.get("SKILL_PATHS", "")
        if skill_paths_str:
            skill_paths = skill_paths_str.split(":")
            skills = discover_skills(skill_paths)
            logger.info(f"Loaded {len(skills)} skills")

    # Create agent context with skills
    agent_context = AgentContext(skills=skills)

    # Parse MCP server config from environment (serialized by openhands_sdk.py)
    mcp_config = None
    mcp_servers_raw = os.environ.get("MCP_SERVERS_JSON")
    if mcp_servers_raw:
        mcp_servers = json.loads(mcp_servers_raw)
        mcp_config = {"mcpServers": {}}
        for mcp in mcp_servers:
            server_name = mcp.get("name", "mcp-server")
            transport = mcp.get("transport", "stdio")
            server_cfg: dict[str, Any] = {}
            if transport == "stdio":
                if mcp.get("command"):
                    server_cfg["command"] = mcp["command"]
                if mcp.get("args"):
                    server_cfg["args"] = mcp["args"]
            else:
                if mcp.get("url"):
                    server_cfg["url"] = mcp["url"]
            mcp_config["mcpServers"][server_name] = server_cfg
        logger.info(f"MCP config: {json.dumps(mcp_config, indent=2)}")

    # Create agent (with optional MCP config)
    agent_kwargs: dict[str, Any] = {
        "llm": llm,
        "tools": tools,
        "agent_context": agent_context,
    }
    if mcp_config:
        agent_kwargs["mcp_config"] = mcp_config
    agent = Agent(**agent_kwargs)

    # Run conversation
    # Use the container's current working directory (set by Dockerfile WORKDIR)
    workspace = os.getcwd()
    conv_kwargs: dict[str, Any] = {"agent": agent, "workspace": workspace}
    max_iter_raw = os.environ.get("MAX_ITERATIONS")
    if max_iter_raw:
        conv_kwargs["max_iteration_per_run"] = int(max_iter_raw)
        logger.info(f"Max iterations per run: {max_iter_raw}")
    conversation = Conversation(**conv_kwargs)

    print(f"Starting agent with instruction: {args.instruction[:200]}...")
    print(f"Using model: {model}")
    if temperature_raw:
        print(f"Temperature: {temperature_raw}")
    if max_iter_raw:
        print(f"Max iterations per run: {max_iter_raw}")
    print(f"Loaded {len(skills)} skills")
    if mcp_config:
        print(f"MCP servers: {list(mcp_config['mcpServers'].keys())}")

    # Send instruction and run
    conversation.send_message(args.instruction)
    conversation.run()

    # Collect metrics from accumulated_token_usage
    token_usage = llm.metrics.accumulated_token_usage
    metrics = {
        "prompt_tokens": token_usage.prompt_tokens if token_usage else 0,
        "completion_tokens": token_usage.completion_tokens if token_usage else 0,
        "cached_tokens": token_usage.cache_read_tokens if token_usage else 0,
        "cost_usd": llm.metrics.accumulated_cost,
    }

    # Extract system prompt and tool definitions from the initialized agent
    system_prompt = None
    tool_definitions: list[dict[str, Any]] = []
    try:
        system_prompt = agent.static_system_message
    except Exception as e:
        logger.warning(f"Could not extract system prompt: {e}")
    try:
        for tool_name, tool_obj in agent.tools_map.items():
            tool_definitions.append(tool_obj.to_openai_tool())
    except Exception as e:
        logger.warning(f"Could not extract tool definitions: {e}")

    if system_prompt:
        print(f"Captured system prompt ({len(system_prompt)} chars)")
    print(f"Captured {len(tool_definitions)} tool definitions")

    # Convert SDK events to dicts for build_trajectory()
    events_list: list[dict[str, Any]] = []
    last_agent_timestamp: str | None = None
    for event in conversation.state.events:
        if isinstance(event, MessageEvent):
            content = ""
            if event.llm_message:
                msg_content = getattr(event.llm_message, "content", None)
                if isinstance(msg_content, list):
                    # Extract text from TextContent objects
                    content = "\n".join(
                        getattr(c, "text", str(c))
                        for c in msg_content
                        if getattr(c, "text", None)
                    )
                elif msg_content:
                    content = str(msg_content)
            if event.source == "user":
                events_list.append(
                    {
                        "type": "user_message",
                        "content": content,
                        "timestamp": event.timestamp,
                    }
                )
            elif event.source == "agent":
                entry: dict[str, Any] = {
                    "type": "assistant_message",
                    "content": content,
                    "timestamp": event.timestamp,
                }
                events_list.append(entry)
                last_agent_timestamp = event.timestamp
        elif isinstance(event, ActionEvent):
            tool_call_args: dict[str, Any] = {}
            # Try tool_call.function.arguments (OpenAI format)
            if event.tool_call and hasattr(event.tool_call, "function"):
                raw_args = getattr(event.tool_call.function, "arguments", None)
                if isinstance(raw_args, str):
                    try:
                        tool_call_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        tool_call_args = {"raw": raw_args}
                elif isinstance(raw_args, dict):
                    tool_call_args = raw_args
            # Fallback: extract from the parsed action's dict representation
            if not tool_call_args and event.action:
                try:
                    action_dict = (
                        event.action.model_dump()
                        if hasattr(event.action, "model_dump")
                        else vars(event.action)
                    )
                    # Remove internal fields
                    tool_call_args = {
                        k: v
                        for k, v in action_dict.items()
                        if k != "kind" and v is not None
                    }
                except Exception:
                    pass
            entry = {
                "type": "assistant_message",
                "content": "",
                "timestamp": event.timestamp,
                "tool_calls": [
                    {
                        "id": event.tool_call_id,
                        "name": event.tool_name,
                        "arguments": tool_call_args,
                    }
                ],
            }
            events_list.append(entry)
            last_agent_timestamp = event.timestamp
        elif isinstance(event, ObservationEvent):
            obs_content = ""
            if event.observation:
                # Try to extract text from observation content
                obs_raw = getattr(event.observation, "content", None)
                if isinstance(obs_raw, list):
                    obs_content = "\n".join(
                        getattr(c, "text", str(c))
                        for c in obs_raw
                        if getattr(c, "text", None)
                    )
                elif obs_raw:
                    obs_content = str(obs_raw)
                else:
                    obs_content = str(event.observation)
            events_list.append(
                {
                    "type": "tool_result",
                    "tool_call_id": event.tool_call_id,
                    "content": obs_content,
                    "timestamp": event.timestamp,
                }
            )
        elif isinstance(event, TokenEvent):
            if last_agent_timestamp and events_list:
                for ev in reversed(events_list):
                    if ev.get("timestamp") == last_agent_timestamp:
                        ev["token_ids"] = {
                            "prompt_token_ids": getattr(event, "prompt_token_ids", []),
                            "response_token_ids": getattr(
                                event, "response_token_ids", []
                            ),
                        }
                        break

    # Build and save trajectory
    trajectory = build_trajectory(
        events_list,
        metrics,
        model,
        system_prompt=system_prompt,
        tool_definitions=tool_definitions,
    )

    trajectory_path = Path(args.trajectory_path)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trajectory_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    raw_trajectory_path = trajectory_path.with_suffix(".raw.json")
    dump_conversation_events(conversation, raw_trajectory_path)
    message_trajectory_path = trajectory_path.with_suffix(".message.json")
    dump_openai_chat_messages(conversation, llm, tool_definitions, message_trajectory_path)

    print(f"Agent completed. Trajectory saved to {trajectory_path}")
    print(f"Total cost: ${metrics['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
