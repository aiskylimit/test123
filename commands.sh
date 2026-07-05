#1
#kill-probe-tests
pkill -f "run_probe_tests.py" 2>/dev/null
pkill -f "test_t[0-9]" 2>/dev/null
pkill -f "test_probe_ab" 2>/dev/null
sleep 3
echo "killed"
