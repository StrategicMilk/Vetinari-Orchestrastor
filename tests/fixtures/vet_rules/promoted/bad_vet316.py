def accepts_claim(path: str, claim_lower: str) -> bool:
    return path.lower() in claim_lower
