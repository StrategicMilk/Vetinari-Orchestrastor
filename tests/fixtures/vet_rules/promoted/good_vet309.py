def certify(ok: bool) -> None:
    if not ok:
        raise RuntimeError("validation failed")
    print("Validation checks completed")
