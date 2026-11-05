"""Sanitized test progress from canonical runner output; no hidden test names escape."""
import hashlib
import shlex
import re


def disable_failfast(script):
    """Disable early stop in simple canonical pytest/go commands in verifier copy.

    Complex shell expressions are left unchanged; coverage is still checked from
    runner output. Original task files and the live workspace are never edited.
    """
    lines = []
    changed = False
    for line in script.splitlines(keepends=True):
        if any(char in line for char in (';', '|', '&', '$', '`', '<', '>', '\\')):
            lines.append(line); continue
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError:
            lines.append(line); continue
        start = None
        runner = None
        if tokens and tokens[0].split('/')[-1] == 'pytest':
            start, runner = 1, 'pytest'
        elif len(tokens) >= 3 and tokens[0].split('/')[-1] in ('python','python3') and tokens[1:3] == ['-m','pytest']:
            start, runner = 3, 'pytest'
        elif tokens[:2] == ['go','test']:
            start, runner = 2, 'go'
        if start is None:
            lines.append(line); continue
        args = tokens[:start]
        i = start
        while i < len(tokens):
            token = tokens[i]
            if runner == 'pytest' and token == '--maxfail':
                i += 2; continue
            if runner == 'pytest' and token.startswith('--maxfail='):
                i += 1; continue
            if runner == 'pytest' and re.fullmatch(r'-[vxqs]+', token) and 'x' in token:
                token = token.replace('x','')
                if token == '-': i += 1; continue
            if runner == 'go' and (token == '-failfast' or token.startswith('-failfast=')):
                i += 1; continue
            args.append(token); i += 1
        if runner == 'pytest': args.append('--maxfail=0')
        if args != tokens:
            changed = True
            lines.append(shlex.join(args) + ('\n' if line.endswith('\n') else ''))
        else: lines.append(line)
    return ''.join(lines), changed


def _count(text, word):
    return sum(int(n) for n in re.findall(r'\b(\d+)\s+' + word + r'\b', text))


def summarize_tests(output, canonical_script, exit_code):
    """Counts refer to executed tests, excluding skips. Missing counts are not zero.

    This intentionally does not parse arbitrary model-written claims. Runner
    families are selected by the canonical script, and require runner summaries.
    The script fingerprint and denominator gate comparisons across rounds.
    """
    fingerprint = hashlib.sha256(canonical_script.encode()).hexdigest()
    base = {'schema_version': 1, 'suite_fingerprint': fingerprint,
            'unit': 'test_case', 'available': False, 'complete': False,
            'observed_pass_rate': None, 'confirmed_pass_fraction': None, 'unverified': None, 'passed': None, 'failed': None,
            'errors': None, 'skipped': None, 'executed': None,
            'selected': None, 'reason': 'No supported canonical test summary'}
    text = re.sub(r'\x1b\[[0-9;]*m', '', output)
    passed = failed = errors = skipped = 0
    complete = False
    selected = None
    if 'pytest' in canonical_script:
        base['runner'] = 'pytest'
        summaries = re.findall(r'^=+\s*(.*?)\s*=+\s*$', text, re.M)
        summaries = [s for s in summaries if re.search(r'\d+ (?:passed|failed|error|skipped|xfailed|xpassed)', s)]
        if len(summaries) != 1:
            return {**base, 'reason': 'Missing or multiple pytest summaries; aggregate scope uncertain'}
        s = summaries[0]
        passed, failed = _count(s, 'passed'), _count(s, 'failed')
        errors = _count(s, 'errors?')
        skipped = _count(s, 'skipped') + _count(s, 'xfailed')
        # XPASS may or may not fail depending on strictness; do not guess.
        if _count(s, 'xpassed'):
            return {**base, 'reason': 'XPASS semantics require a structured test report'}
        collected = re.findall(r'collected (\d+) items?(?:\s*/\s*(\d+) deselected\s*/\s*(\d+) selected)?', text)
        if len(collected) == 1:
            n, deselected, chosen = collected[0]
            selected = int(chosen) if chosen else int(n)
        total = passed + failed + errors + skipped
        complete = selected is not None and total == selected and not re.search(r'stopping after|Interrupted|ERROR collecting|INTERNALERROR', text)
    elif re.search(r'\bgo\s+test\b', canonical_script):
        base['runner'] = 'go'
        # Only top-level verdicts: indented subtests must not be counted twice.
        verdicts = re.findall(r'^--- (PASS|FAIL|SKIP): (\S+)', text, re.M)
        if not verdicts:
            return base
        passed = sum(v == 'PASS' for v, _ in verdicts)
        failed = sum(v == 'FAIL' for v, _ in verdicts)
        skipped = sum(v == 'SKIP' for v, _ in verdicts)
        runs = re.findall(r'^=== RUN\s+(\S+)', text, re.M)
        top_runs = [n for n in runs if '/' not in n]
        package_results = re.findall(r'^(?:ok\s+\S+|FAIL\s+\S+)', text, re.M)
        complete = bool(package_results) and len(top_runs) == len(verdicts) and not re.search(r'\[build failed\]|panic:|fatal error:|\[setup failed\]', text)
        if '-failfast' in canonical_script and failed:
            complete = False
        selected = len(verdicts) if complete else None
        # Runtime-discovered names identify the evaluated scope without exposing it.
        base['suite_fingerprint'] = hashlib.sha256((canonical_script + '\n' + '\n'.join(sorted(n for _, n in verdicts))).encode()).hexdigest()
    elif 'ctest' in canonical_script:
        base['runner'] = 'ctest'
        matches = re.findall(r'(\d+)% tests passed, (\d+) tests failed out of (\d+)', text)
        if len(matches) != 1:
            return base
        _, fail, total = matches[0]
        failed, selected = int(fail), int(total)
        # CTest's percentage includes skipped/not-run cases; do not call them passes.
        if re.search(r'\*\*\*Skipped|\*\*\*Not Run|DID NOT RUN|tests did not run', text, re.I):
            return {**base, 'reason': 'CTest skipped/not-run tests require structured results'}
        passed = selected - failed
        complete = True
    elif 'unittest' in canonical_script:
        base['runner'] = 'unittest'
        runs = re.findall(r'^Ran (\d+) tests? in ', text, re.M)
        outcomes = re.findall(r'^(OK(?: \([^\n]*\))?|FAILED \([^\n]*\))$', text, re.M)
        if len(runs) != 1 or len(outcomes) != 1:
            return base
        s = outcomes[0]
        def field(name):
            match = re.search(name + r'=(\d+)', s)
            return int(match[1]) if match else 0
        failed, errors = field('failures'), field('errors')
        skipped = field('skipped') + field('expected failures')
        if field('unexpected successes'):
            return {**base, 'reason': 'Unexpected success semantics require structured results'}
        selected = int(runs[0]);passed = selected - failed - errors - skipped
        complete = not (('-f' in canonical_script or '--failfast' in canonical_script) and failed + errors)
        # unittest's "Ran N" counts executed tests, not the selected suite.
        # After failfast it cannot serve as the task's total denominator.
        if not complete:
            selected = None
    else:
        return base
    executed = passed + failed + errors
    if exit_code >= 124 or exit_code < 0:
        complete = False
    if executed <= 0 or passed < 0:
        return {**base, 'reason': 'No reliably counted executed tests'}
    return {**base, 'available': True, 'passed': passed, 'failed': failed,
            'errors': errors, 'skipped': skipped, 'executed': executed,
            'selected': selected, 'complete': complete,
            'observed_pass_rate': passed / executed,
            'confirmed_pass_fraction': passed / selected if selected else None,
            'unverified': max(0, selected - executed) if selected is not None else None,
            'reason': 'Complete canonical test run' if complete else 'Partial or unconfirmed test coverage; not a whole-suite pass rate'}


def feedback_history(rounds):
    """Expose numeric outcomes and aggregate counts, never names or raw logs."""
    history = []
    previous = None
    for row in rounds:
        progress = row.get('test_progress') or {}
        rate = progress.get('confirmed_pass_fraction')
        comparable = bool(previous and rate is not None and previous.get('confirmed_pass_fraction') is not None and progress.get('available') and previous.get('available')
            and progress.get('complete') and previous.get('complete')
            and progress.get('suite_fingerprint') == previous.get('suite_fingerprint')
            and progress.get('executed') == previous.get('executed')
            and progress.get('selected') == previous.get('selected')
            and progress.get('skipped') == previous.get('skipped'))
        delta = rate - previous['confirmed_pass_fraction'] if comparable else None
        history.append({'round_id': row['round_id'], 'functional_reward': row['functional_reward'],
            'test_progress': {k: progress.get(k) for k in ('available','runner','unit','confirmed_pass_fraction','unverified',
                'passed','failed','errors','skipped','executed','selected','complete','reason')},
            'comparable_to_previous': comparable, 'confirmed_pass_fraction_delta': delta})
        previous = progress
    return {'outcome': 'passed' if rounds[-1]['functional_reward'] == 1 else 'not_passed',
            'functional_reward': rounds[-1]['functional_reward'], 'history': history} if rounds else {}
