"""Optional job snapshots; ordinary two-argument progress callbacks still work."""


def publish(progress, details):
    callback = getattr(progress, "snapshot", None)
    if callback is not None:
        callback(details)
