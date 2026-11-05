from harbor_patch.multiturn.test_progress import summarize_tests, feedback_history, disable_failfast


def test_fixed_denominator_after_first_failure():
    p=summarize_tests('collected 10 items\n!!! stopping after 1 failures !!!\n=== 1 failed, 2 passed in 1s ===\n','pytest -xvs tests',1)
    assert p['confirmed_pass_fraction']==.2
    assert p['executed']==3 and p['unverified']==7
    assert not p['complete']


def test_fixed_denominator_after_five_executed():
    p=summarize_tests('collected 10 items\n!!! stopping after 3 failures !!!\n=== 3 failed, 2 passed in 1s ===\n','pytest tests',1)
    assert p['confirmed_pass_fraction']==.2
    assert p['executed']==5 and p['unverified']==5


def test_full_suite_progress_and_safe_delta():
    script='pytest tests --maxfail=0'
    p=summarize_tests('collected 10 items\n=== 8 failed, 2 passed in 1s ===\n',script,1)
    q=summarize_tests('collected 10 items\n=== 3 failed, 7 passed in 1s ===\n',script,1)
    h=feedback_history([{'round_id':1,'functional_reward':0,'test_progress':p}, {'round_id':2,'functional_reward':0,'test_progress':q}])
    assert h['history'][1]['test_progress']['confirmed_pass_fraction']==.7
    assert abs(h['history'][1]['confirmed_pass_fraction_delta']-.5)<1e-10
    assert 'suite_fingerprint' not in str(h)
    assert 'observed_pass_rate' not in str(h)
    q['selected']=11
    assert feedback_history([{'round_id':1,'functional_reward':0,'test_progress':p},{'round_id':2,'functional_reward':0,'test_progress':q}])['history'][1]['confirmed_pass_fraction_delta'] is None


def test_skip_remains_unverified_not_passed():
    p=summarize_tests('collected 10 items\n=== 2 passed, 8 skipped in 1s ===\n','pytest tests',0)
    assert p['confirmed_pass_fraction']==.2 and p['unverified']==8


def test_missing_denominator_is_not_invented():
    p=summarize_tests('=== 2 passed in 1s ===\n','pytest tests',0)
    assert p['confirmed_pass_fraction'] is None
    assert not p['complete']


def test_go_does_not_double_count_subtests_or_expose_names():
    text='=== RUN   TestSecret\n=== RUN   TestSecret/Child\n    --- PASS: TestSecret/Child (0s)\n--- PASS: TestSecret (0s)\nok  example 0.1s\n'
    p=summarize_tests(text,'go test -v ./...',0)
    assert p['passed']==1 and p['selected']==1 and p['confirmed_pass_fraction']==1
    assert 'TestSecret' not in str(feedback_history([{'round_id':1,'functional_reward':1,'test_progress':p}]))


def test_unsupported_or_build_failure_is_not_zero_progress():
    assert not summarize_tests('build failed','go test ./...',1)['available']
    assert not summarize_tests('all good','custom_runner',0)['available']


def test_ctest_and_unittest():
    assert summarize_tests('20% tests passed, 8 tests failed out of 10','ctest',8)['confirmed_pass_fraction']==.2
    assert summarize_tests('Ran 10 tests in 0.5s\nFAILED (failures=8)\n','python -m unittest',1)['confirmed_pass_fraction']==.2


def test_unittest_failfast_does_not_use_executed_count_as_total():
    p=summarize_tests('Ran 3 tests in 0.5s\nFAILED (failures=1)\n',
                      'python -m unittest --failfast',1)
    assert p['passed']==2 and p['executed']==3
    assert p['selected'] is None and p['confirmed_pass_fraction'] is None
    assert not p['complete']


def test_disable_early_stop_preserves_selection():
    script='source /opt/venv/bin/activate\npytest -xvs tests/a.py\ngo test -v -failfast ./pkg\n'
    prepared, changed=disable_failfast(script)
    assert changed and 'pytest -vs tests/a.py --maxfail=0' in prepared
    assert '-failfast' not in prepared
    assert 'source /opt/venv/bin/activate' in prepared


import pytest

@pytest.mark.parametrize('total', [4, 10, 25])
@pytest.mark.parametrize('stop_early', [False, True])
def test_real_pytest_uses_each_tasks_total(tmp_path, total, stop_early):
    import subprocess
    import sys
    import shlex
    path = tmp_path / 'test_task.py'
    path.write_text(f'import pytest\n@pytest.mark.parametrize("i", range({total}))\ndef test_case(i):\n    assert i < 2\n')
    command = shlex.join([sys.executable, '-m', 'pytest', '-xvs', str(path)])
    if not stop_early:
        command, changed = disable_failfast(command)
        assert changed
    result = subprocess.run(command, shell=True, text=True, capture_output=True, timeout=30)
    progress = summarize_tests(result.stdout + result.stderr, command, result.returncode)
    assert progress['selected'] == total
    assert progress['passed'] == 2
    assert progress['confirmed_pass_fraction'] == 2 / total
    assert progress['executed'] == (3 if stop_early else total)
    assert progress['unverified'] == (total - 3 if stop_early else 0)
    assert progress['complete'] is (not stop_early)
