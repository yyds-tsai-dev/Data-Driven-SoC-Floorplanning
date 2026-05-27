## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Code Search

Use `semble search` to find code by describing what it does or naming a symbol/identifier, instead of grep:

```bash
semble search "authentication flow" .
semble search "save_pretrained" .
semble search "save model to disk" . --top-k 10
```

If you anticipate doing more than one search, use `semble index` to create an index:

```bash
semble index . -o .semble-index
semble search "save_pretrained" --index .semble-index
```

Use `--content docs` to search documentation and prose, `--content config` for config files, or `--content all` to search code, docs, and config:

```bash
semble search "deployment guide" . --content docs
semble search "database host port" . --content config
semble search "authentication" . --content all
```

Use `semble find-related` to discover code similar to a known location:

```bash
semble find-related src/auth.py 42 .
```

An index is not automatically updated, so reindex after significant code changes or if results look stale. Use grep only when you need exhaustive literal matches or quick confirmation of an exact string.
