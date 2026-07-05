#1 +20+a
#verify-clean
# Kill any remaining probe processes
pkill -f "run_probe_tests.py" 2>/dev/null
pkill -f "test_t[0-9]" 2>/dev/null
pkill -f "test_probe_ab" 2>/dev/null
sleep 5

# Check GPU state
nvidia-smi

# Remove partial results
rm -rf /opt/dlami/nvme/probe_results
echo "probe_results removed"

# Verify no python processes on GPUs
nvidia-smi --query-compute-apps=pid,name --format=csv,noheader
echo "done"
