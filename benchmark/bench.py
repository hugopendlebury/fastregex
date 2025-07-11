import re  # noqa: D100
import time
from random import choices
from string import ascii_letters, digits
from fastre import (
    compile,
    escape,
    fmatch,
    findall,
    #finditer,
    fullmatch,
    search,
    split,
    sub,
    subn,
)

import re

def benchmark(func, *args, iterations=1000):  # noqa: D103
    start = time.time()
    for _ in range(iterations):
        func(*args)
    end = time.time()
    return (end - start) * 1000  # Convert to milliseconds

# Test cases
PATTERN = r'(\w+)\s+(\d+)'
TEXT = ''.join(choices(ascii_letters + digits, k=1000))
ITERATIONS = 100
EMAIL_PATTERN = r"^(?!\.)(?!.*\.\.)([a-z0-9_'+\-\.]*)[a-z0-9_'+\-]@([a-z0-9][a-z0-9\-]*\.)+[a-z]{2,}$"
BACKTRACK_TEXT = "ababababababababababababababab"
BACKREF = "hello there this is a test of a back reference abcd with some extra text after"

if __name__ == "__main__":
    fastre_results = [
        ("Compile", benchmark(compile, PATTERN, iterations=ITERATIONS), True),
        ("Search", benchmark(search, compile(PATTERN), TEXT, iterations=ITERATIONS), True),
        ("Find Match", benchmark(fmatch, compile(PATTERN), TEXT, iterations=ITERATIONS), True),
        ("Full Match", benchmark(fullmatch, compile(PATTERN), TEXT, iterations=ITERATIONS), True),
        ("Split", benchmark(split, compile(PATTERN), TEXT, iterations=ITERATIONS), True),
        ("Find All", benchmark(findall, compile(PATTERN), TEXT, iterations=ITERATIONS), True),
       # ("Find Iter", benchmark(finditer, compile(PATTERN), TEXT, iterations=ITERATIONS), True),
        ("Sub", benchmark(sub, compile(PATTERN), 'foo', TEXT, iterations=ITERATIONS), True),
        ("Subn", benchmark(subn, compile(PATTERN), 'foo', TEXT, iterations=ITERATIONS), True),
        ("Escape", benchmark(escape, TEXT, iterations=ITERATIONS), False),
        ("Email", benchmark(fmatch, compile(EMAIL_PATTERN), "valid@domain.com", iterations=10), True),
        ("backtrack", benchmark(fmatch, compile("(?i)(a|b|ab)*(?=c)"), BACKTRACK_TEXT, iterations=1), True),
        ("backreference", benchmark(fmatch, compile(".*?a(bc(?=d)|b)cd"), BACKREF, iterations=1), True),
        ("backref2", benchmark(fmatch, compile("foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo"), r"(\w+) (\1)", iterations=ITERATIONS), True)
    ]

    re_results = [
        ("Compile", benchmark(re.compile, PATTERN, iterations=ITERATIONS), True),
        ("Search", benchmark(lambda pattern: pattern.search(TEXT), re.compile(PATTERN), iterations=ITERATIONS), True),
        ("Find Match", benchmark(lambda pattern: pattern.match(TEXT), re.compile(PATTERN), iterations=ITERATIONS), True),
        ("Full Match", benchmark(lambda pattern: pattern.fullmatch(TEXT), re.compile(PATTERN), iterations=ITERATIONS), True),
        ("Split", benchmark(lambda pattern: pattern.split(TEXT), re.compile(PATTERN), iterations=ITERATIONS), True),
        ("Find All", benchmark(lambda pattern: pattern.findall(TEXT), re.compile(PATTERN), iterations=ITERATIONS), True),
       # ("Find Iter", benchmark(lambda pattern: list(pattern.finditer(TEXT)), re.compile(PATTERN), iterations=ITERATIONS), True),
        ("Sub", benchmark(lambda pattern: pattern.sub('foo', TEXT), re.compile(PATTERN), iterations=ITERATIONS), True),
        ("Subn", benchmark(lambda pattern: pattern.subn('foo', TEXT), re.compile(PATTERN), iterations=ITERATIONS), True),
        ("Escape", benchmark(re.escape, TEXT, iterations=ITERATIONS), False),
        ("Email", benchmark(lambda pattern: pattern.match("valid@domain.com"), re.compile(EMAIL_PATTERN), iterations=10), True),
        ("backtrack", benchmark(lambda pattern: pattern.match(BACKTRACK_TEXT), re.compile("(?i)(a|b|ab)*(?=c)"), iterations=1), True),
        ("backreference", benchmark(lambda pattern: pattern.match(BACKREF), re.compile("(?i)(a|b|ab)*(?=c)"), iterations=1), True),
        ("backref2", benchmark(lambda pattern: pattern.match("foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo"), re.compile(r"(\w+) (\1)"), iterations=ITERATIONS), True)
    ]

    max_len = max(len(op) for op, _, _ in fastre_results + re_results)
    print(f"{'Operation':{max_len}} | {'fastre (ms)':<10} | {'re (ms)':<10} | {'Used Regex':<10} | {'Faster':<10}")
    print("-" * (max_len + 45))

    fastre_times = []
    re_times = []

    for (op, fastre_time, used_regex_flpc), (_, re_time, used_regex_re) in zip(fastre_results, re_results):
        used_regex = "Yes" if used_regex_flpc and used_regex_re else "No"
        faster = "fastre" if fastre_time < re_time else "re"
        print(f"{op:{max_len}} | {fastre_time:<10.5f} | {re_time:<10.5f} | {used_regex:<10} | {faster:<10}")
        fastre_times.append(fastre_time)
        re_times.append(re_time)

    mean_flpc = sum(fastre_times) / len(fastre_times)
    mean_re = sum(re_times) / len(re_times)
    performance_ratio = mean_re / mean_flpc

    print(f"\nfastre is {performance_ratio:.2f}x faster than re module on average")