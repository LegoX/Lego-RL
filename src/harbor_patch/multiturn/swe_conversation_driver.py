"""Inside the sandbox: preserved CC session, ordinary user turns, no review gate."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import urllib.error
import urllib.request
import uuid


SYSTEM = '''Work with the user on their coding task. Respond to their questions and
corrections in the same conversation. Ask a concise question when necessary;
otherwise inspect, implement and verify the work. A final text response ends the
current user turn. Further user feedback may follow. No maintainer approval step
or special validation tool is required. Run appropriate checks with ordinary
tools and describe their actual results. Do not invent successful validation.
Use ordinary assistant text to ask questions; AskUserQuestion is unavailable.
Work from the current repository, documentation and tests. Do not retrieve prior
solutions from Git history, remotes or the web. Do not modify benchmark harness
files or anything mounted under /tests.
'''


class ContextLimitReached(RuntimeError):
    """A structured CLI context-limit result, eligible for final verification."""


def query_with_context_limit(query, result_log):
    try:
        return query()
    except RuntimeError as exc:
        # Check this invocation's CLI record, not simulator/network error text.
        if not str(exc).startswith('ClaudeCodeExecutionError:'):
            raise
        result = None
        try:
            with result_log.open() as stream:
                for line in stream:
                    value = json.loads(line)
                    if value.get('type') == 'result':
                        result = value
        except (OSError, ValueError):
            raise exc
        if (result and result.get('is_error') is True
                and result.get('result') == 'Prompt is too long'
                and result.get('terminal_reason') == 'blocking_limit'):
            raise ContextLimitReached('CLI context budget exhausted') from exc
        raise


def drive(query, exchange, max_iterations):
    response = exchange({'round': 0})
    remaining = max_iterations
    number = 0
    while response.get('action') == 'reply':
        if response.get('round') != number + 1:
            raise ValueError('Nonconsecutive user round')
        number += 1
        message, used = query(response['message'], number, remaining)
        remaining -= used
        response = exchange({'round': number, 'assistant_message': message,
                             'budget_exhausted': remaining <= 0})
    if response.get('action') != 'stop' or response.get('reason') not in ('user_done', 'max_rounds', 'max_iterations', 'verifier_success'):
        raise RuntimeError('Invalid conversation controller response')
    return response['reason'], number


def exchange_conversation(endpoint, payload, timeout_sec):
        request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                # Only propagate our structured error code, never arbitrary bodies.
                if exc.code == 503:
                    try:
                        error = json.loads(exc.read(4096)).get('error', {})
                    except (ValueError, AttributeError):
                        error = {}
                    if isinstance(error, dict) and error.get('code') == 'conversation_turn_failed':
                        raise RuntimeError('Conversation turn failed: controller rejected replay') from None
                    if isinstance(error, dict) and error.get('code') == 'simulator_unavailable':
                        raise RuntimeError('User simulator unavailable: retries exhausted') from None
                if exc.code not in (502, 504) or attempt == 2:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 2:
                    raise
            time.sleep(attempt + 1)


def main():
    # The pinned runtime enables Python safe-path mode.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_claude_harbor as cli
    # Isolated driver process, never a shared RL worker.
    cli.SYSTEM = SYSTEM
    config = dict(os.environ)
    endpoint = config.pop('CONVERSATION_URL')
    os.environ.pop('CONVERSATION_URL', None)
    root = Path(config['CUSTOM_AGENT_RUNTIME_ROOT'])
    environment = cli.clean_environment(config, root)
    os.environ.clear()
    os.environ.update(environment)
    logs = Path('/logs/agent')
    logs.mkdir(parents=True, exist_ok=True)
    sessions = logs / 'sessions'
    sessions.mkdir(exist_ok=True)
    (sessions / '.claude.json').write_text(json.dumps({'hasCompletedOnboarding': True}))
    settings = logs / 'settings.json'
    settings.write_text('{}')
    environment.update({
        'ANTHROPIC_BASE_URL': config['ANTHROPIC_BASE_URL'], 'ANTHROPIC_API_KEY': 'local-policy',
        'ANTHROPIC_CUSTOM_HEADERS': '', 'CLAUDE_CONFIG_DIR': str(sessions),
        'IS_SANDBOX': '1', 'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
        'DISABLE_AUTOUPDATER': '1', 'CLAUDE_CODE_DISABLE_BACKGROUND_TASKS': '1',
        'CLAUDE_CODE_DISABLE_AUTO_MEMORY': '1', 'DISABLE_AUTO_COMPACT': '1', 'DISABLE_COMPACT': '1',
        'CLAUDE_CODE_MAX_OUTPUT_TOKENS': '4096', 'HARBOR_CC_EFFORT': 'high',
        'CLAUDE_CODE_MAX_CONTEXT_TOKENS': config.get('CLAUDE_CODE_MAX_CONTEXT_TOKENS', '64000'),
    })
    for key in ('ANTHROPIC_MODEL', 'ANTHROPIC_DEFAULT_SONNET_MODEL', 'ANTHROPIC_DEFAULT_OPUS_MODEL',
                'ANTHROPIC_DEFAULT_HAIKU_MODEL', 'CLAUDE_CODE_SUBAGENT_MODEL'):
        environment[key] = config['CC_MODEL']
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    def exchange(payload):
        return exchange_conversation(endpoint, payload, float(config.get('CONVERSATION_TIMEOUT_SEC', '600')))
    session_id = str(uuid.uuid4())
    status = {'session_id': session_id, 'interaction_mode': 'conversation'}
    def query(prompt, turn, remaining):
        status['interaction_turns'] = turn
        return query_with_context_limit(
            lambda: cli.run_cli(prompt, cli=root / 'bin/claude',
                session_id=session_id, resume=turn > 1, settings=settings,
                environment=environment, logs=logs, turn=turn, remaining=remaining),
            logs / f'claude-turn-{turn:03d}.jsonl')
    try:
        reason, rounds = drive(query, exchange, int(config.get('MAX_ITERATIONS', '500')))
        status.update(stop_reason=reason, interaction_turns=rounds)
    except ContextLimitReached:
        status.update(stop_reason='context_limit', context_limit_source='claude_cli')
    except BaseException as exc:
        status.update(error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        (logs / 'interaction-result.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
