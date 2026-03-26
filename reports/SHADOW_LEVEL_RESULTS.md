# Shadow Level LLM Comprehension — Test Report

**Date:** 2026-03-26
**Models:** phi3:mini (3.8B), gemma2:2b, llama3.2:1b
**Environment:** ollama 0.18.2, localhost, temperature=0, 3 runs per test

## Summary

L0 (fields only) is the optimal presentation for lightweight LLMs. Protocol information ($S header, schema ID) is noise at this model size.

## Shadow Levels Tested

```
L0: fields only       [service, status, score, uptime_hrs]
L2: full protocol     ["$S","service-status:v1",4,"service","status","score","uptime_hrs"]
L4: NL fallback       service: api-gateway, status: healthy, score: 0.95, uptime_hrs: 720
```

## Results

| Model | Test | L0 (fields) | L2 (full $S) | L4 (NL) |
|-------|------|:-----------:|:------------:|:-------:|
| phi3:mini | field_lookup | **3/3** | 3/3 | 3/3 |
| phi3:mini | count_filter | **3/3** | 3/3 | 3/3 |
| phi3:mini | max_value | **3/3** | 0/3 | 0/3 |
| gemma2:2b | field_lookup | 3/3 | 3/3 | 3/3 |
| gemma2:2b | count_filter | 0/3 | 0/3 | 0/3 |
| gemma2:2b | max_value | 0/3 | **3/3** | 0/3 |
| llama3.2:1b | field_lookup | **3/3** | 0/3 | 3/3 |
| llama3.2:1b | count_filter | 3/3 | 3/3 | 3/3 |
| llama3.2:1b | max_value | 0/3 | 0/3 | 0/3 |

### Totals

| Model | L0 | L2 | L4 |
|-------|:--:|:--:|:--:|
| **phi3:mini** | **9/9** | 6/9 | 6/9 |
| gemma2:2b | 3/9 | **6/9** | 3/9 |
| llama3.2:1b | **6/9** | 3/9 | 6/9 |

## Findings

1. **L0 is best for phi3 and llama.** Protocol information (`$S`, schema ID, field count) adds no value and actively hurts comprehension on max_value task. Fields-only header lets the model focus on data.

2. **gemma2 is the exception** — L2 outperforms L0 on max_value. Hypothesis: gemma2 benefits from the structured `$S` array format as a parsing anchor. However, it still fails count_filter across all levels (consistent with prior test — numeric tasks are gemma2's weakness).

3. **L4 (NL) offers no advantage over L0.** Natural language key-value pairs are not easier to parse than a field-name header + positional arrays. NL fallback is a last resort, not an optimization.

4. **phi3:mini at ~3.8B is the practical floor.** 9/9 on L0 across all task types. Below this size, models fail on task complexity regardless of presentation format.

## Implications for Shadow Level Design

- **Default for lightweight agents (≤4B):** L0 (fields only)
- **Default for capable agents (7B+):** L2 (full protocol) — they need schema ID for multi-schema sessions
- **L4 NL fallback:** only for non-DCP-aware consumers, not as an optimization
- **Gateway adaptive logic:** start at L0, promote to L2 when agent demonstrates multi-schema competence

## Raw Data

- [shadow_level_results.json](shadow_level_results.json) — Full test results (3 models × 3 tests × 3 levels)