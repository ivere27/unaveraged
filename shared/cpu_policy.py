"""CPU budget for local model preparation; independent of native dependencies."""
import os
import re


def available_cpu_count():
    try:
        count = len(os.sched_getaffinity(0))
        if count:
            return count
    except (AttributeError, OSError, NotImplementedError):
        pass
    process_count = getattr(os, 'process_cpu_count', os.cpu_count)
    return max(1, process_count() or 1)


def resolve_cpu_threads(value=None):
    """Auto reserves two available logical CPUs, retaining at least one worker."""
    if value is None:
        value = os.environ.get('JOBS', 'auto')
    if isinstance(value, str):
        value = value.strip().lower()
        if value == 'auto':
            return max(1, available_cpu_count() - 2)
        if re.fullmatch(r'[1-9][0-9]*', value):
            return int(value)
    elif type(value) is int and value > 0:
        return value
    raise ValueError('JOBS / --cpu-threads must be auto or a positive integer')
