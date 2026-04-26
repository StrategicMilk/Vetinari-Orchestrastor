import re


def promote(fields, candidates):
    compiled = re.compile(fields["pattern"])
    candidates.append({"pattern": compiled.pattern})
