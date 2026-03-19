import os

# Named output directory for all DLIO benchmark tests.
# Prevents DLIO from creating an ambiguous 'output/' folder in the working
# directory.  Override with DLIO_TEST_OUTPUT_DIR before running pytest:
#   DLIO_TEST_OUTPUT_DIR=my_results pytest dlio_benchmark/tests/
_DEFAULT_TEST_OUTPUT = 'dlio_test_output'
DLIO_TEST_OUTPUT_DIR = os.environ.get('DLIO_TEST_OUTPUT_DIR', _DEFAULT_TEST_OUTPUT)

# Set DLIO_OUTPUT_FOLDER so that _apply_env_overrides() in config.py picks it
# up for every DLIOBenchmark instantiation, including standalone calls that
# don't go through a test-local run_benchmark() wrapper.
os.environ.setdefault('DLIO_OUTPUT_FOLDER', DLIO_TEST_OUTPUT_DIR)


# HACK: to fix the reinitialization problem
def pytest_configure(config):
    config.is_dftracer_initialized = False
