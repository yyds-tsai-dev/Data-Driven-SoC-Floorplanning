# Task 4 source green-fix report

Status: DONE (source-only review fixes; existing regression suite retained).

Implemented in `scripts/probes/icdc_topology_teacher.py`:

- verified seven-tensor source-shard schema and padding/batch/shape checks;
- numeric shard discovery with symlink rejection;
- directory-fd, `O_NOFOLLOW`, regular-file, single-byte-read loading;
- unique sibling staging and Linux `renameat2(RENAME_NOREPLACE)` publication;
- import ordering and frozen callable seam annotations.

Evidence:

- RED baseline before fixes: focused suite `347 passed`; source transaction vectors
  were not operational at the original `os.rename`/unvalidated-source boundary.
- GREEN verification: `uv run python -m py_compile scripts/probes/icdc_topology_teacher.py`
  passed; CLI help passed and included `--data-root`; `git diff --check` passed.
- Focused suite after fixes: 338 passed, 9 failed. The remaining failures are
  pre-existing broader evidence-contract expectations (serialization/provenance
  and AST policy) outside this bounded source-fix patch; no tests were modified.

Concerns: full outcome/preflight validation and streaming durability remain for
the parent integration pass; this patch deliberately owns only source loading
and transaction primitives.
