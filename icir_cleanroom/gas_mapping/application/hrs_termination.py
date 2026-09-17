"""Method-independent termination rules; only real concentration and counts."""


def attempts_exhausted(attempts, limit):
    return attempts >= limit
