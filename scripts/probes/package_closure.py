import re, sys, os

import os as _os
ROOT = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "src", "solver")
IGNORE = {"iccad2026_evaluate"}

seen = set()
queue = ["contest_optimizer"]

import_re = re.compile(r'^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.,\s]+))')

def module_exists(mod):
    top = mod.split('.')[0]
    return os.path.exists(os.path.join(ROOT, top + ".py"))

def extract_modules(line):
    m = import_re.match(line)
    if not m:
        return []
    if m.group(1):
        return [m.group(1).split('.')[0]]
    else:
        mods = m.group(2).split(',')
        return [x.strip().split(' ')[0].split('.')[0] for x in mods if x.strip()]

while queue:
    cur = queue.pop()
    if cur in seen or cur in IGNORE:
        continue
    path = os.path.join(ROOT, cur + ".py")
    if not os.path.exists(path):
        continue
    seen.add(cur)
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            for mod in extract_modules(line):
                if mod in IGNORE:
                    continue
                if module_exists(mod) and mod not in seen:
                    queue.append(mod)

seen.discard("contest_optimizer")
for m in sorted(seen):
    print(m)
