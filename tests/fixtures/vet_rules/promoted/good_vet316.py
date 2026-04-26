def accepts_claim(path: str, evidence_lower: str) -> bool:
    return path.lower() in evidence_lower
