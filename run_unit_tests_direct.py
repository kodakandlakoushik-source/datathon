"""
Standalone test execution script running all tests directly.
"""
import sys
import os
import traceback

# Ensure root dir in path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

print("=" * 60)
print(f"CIVICPULSE AI TEST SUITE — Python {sys.version.split()[0]}")
print("=" * 60)

passed = 0
failed = 0
tests_to_run = []

# Import test modules
import tests.test_priority_engine as t_prio
import tests.test_resource_planner as t_res
import tests.test_data_loader as t_data
import tests.test_ml_engine as t_ml

for mod in [t_prio, t_res, t_data, t_ml]:
    for attr in dir(mod):
        if attr.startswith("test_"):
            tests_to_run.append((mod.__name__, attr, getattr(mod, attr)))

print(f"Discovered {len(tests_to_run)} unit test cases across 4 test suites.\n", flush=True)

for mod_name, test_name, test_fn in tests_to_run:
    try:
        test_fn()
        print(f"  [PASS] {mod_name}.{test_name}", flush=True)
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {mod_name}.{test_name} -> {e}", flush=True)
        traceback.print_exc()
        failed += 1

print("\n" + "=" * 60, flush=True)
print(f"SUMMARY: {passed} PASSED | {failed} FAILED | Total: {passed + failed}", flush=True)
print("=" * 60, flush=True)

if failed > 0:
    sys.exit(1)
sys.exit(0)
