# Evidence — Day 22 Lab Results

## RAGAS Evaluation: V1 vs V2 (Final, Optimized)

| Metric | V1 (Concise) | V2 (Structured) | Winner |
|--------|:-----------:|:---------------:|:-----:|
| Faithfulness | 0.9114 ⭐ | **0.9567** ⭐ | V2 |
| Answer Relevancy | 0.8977 | 0.8930 | V1 |
| Context Recall | **1.0000** | **1.0000** | Tie |
| Context Precision | 0.8241 | 0.8231 | V1 |

✅ Both versions exceed the 0.8 target and achieve the 0.9 bonus threshold.

### V2 Optimization

Initial V2 faithfulness was 0.7869. Two improvements pushed it to 0.9567:

1. **Prompt rewrite**: Changed V2 from "write 3-5 detailed sentences" to "present facts from context, do NOT add information beyond the context." This forced V2 to ground every claim in retrieved documents.
2. **Increased retriever k from 3 to 5**: More context chunks → better coverage → fewer ungrounded claims.

After optimization, V2 actually outperforms V1 on faithfulness while maintaining comparable relevancy. Context recall hit 1.0 for both versions with k=5.

### Routing Distribution

MD5 hash-based deterministic routing: V1=19, V2=31 out of 50 queries.
