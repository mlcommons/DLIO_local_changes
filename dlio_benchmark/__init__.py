# boto3/botocore are banned — block immediately on dlio_benchmark import.
try:
    from mlpstorage.ban_boto3 import install as _ban_boto3
    _ban_boto3()
except ImportError:
    pass  # mlpstorage not installed in this env; skip gracefully
