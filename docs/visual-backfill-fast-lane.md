# Visual backfill and fast-lane evidence

## Safety boundaries

- `rag_app.py visuals` never invokes a document parser. It requires a
  source-matching accepted `parser_result.json` and `parser_cache.json`.
- Successful visual results are content-addressed and immediately durable.
  Failed and budget-deferred element IDs keep the manifest incomplete.
- `benchmark-ingestion` writes only below its output directory. It never opens
  or writes the production Chroma collection.
- The fast lane is disabled by default.

## Real benchmark results

### Mixed 39-page Fishpot report

- Quality: page coverage 1.000, token recall 0.983459, table-page recall 1.000,
  semantic-title-page recall 1.000.
- Routing: 8 native pages, 31 Docling pages.
- Candidate: 18.38 seconds.
- Same-run full Docling: 17.15 seconds.
- Speedup: 0.93271x.
- Decision: rejected by `minimum_speedup`.

### AbraSilver pages 40–62

- Quality: page coverage 1.000, token recall 0.994455, character recall
  0.962956, table-page recall 1.000, semantic-title-page recall 1.000.
- Routing: 18 native pages, 5 Docling pages.
- Candidate: 15.28 seconds.
- Same-run full Docling: 25.58 seconds.
- Speedup: 1.673665x.
- Decision: passes all current quality and performance gates.

These results show that the fast lane is beneficial for long consecutive
simple-page runs but harmful for heavily interleaved layouts. The minimum
native-window policy is therefore part of the production interface, and the
feature remains opt-in pending a larger representative benchmark set.
