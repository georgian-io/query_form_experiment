You are auditing whether a rewritten search query still asks for the same thing as the original.

Judge the INFORMATION NEED, not the wording. The test is the set of documents that correctly
answer the query: a rewrite preserves intent if and only if that set is unchanged — every
document that answers the original still answers the rewrite, and no document that fails the
original now answers the rewrite.

Mark **DRIFTED** if the rewrite does any of these:
- adds, drops, or changes a constraint that narrows or widens which documents qualify — a named
  entity, condition, quantity, unit, timeframe, location, or qualifier (e.g. dropping "in adults",
  "after surgery", "since 2020", or a specific drug/law/function name);
- generalizes a specific need into a broader one, or narrows a broad need into a specific one;
- changes the task or the type of answer being sought (e.g. from "how to treat" to "what causes").

Mark **PRESERVED** if the rewrite only changes wording, phrasing, length, register, or how
specifically the need is expressed, while the set of correct answers is unchanged. A short,
keyword-style rewrite that keeps every answer-determining term is PRESERVED — brevity alone is not
drift. Re-ordering, synonyms, and dropped filler ("I was wondering if you could tell me…") are
PRESERVED.

Mark **UNCERTAIN** only when the ORIGINAL query is itself too vague or garbled to judge what it
asks for. Do not use UNCERTAIN for a rewrite you find merely awkward.

Original query:
{original}

Rewritten query:
{transformed}

Answer with a single JSON object and nothing else:
{{"verdict": "PRESERVED" | "DRIFTED" | "UNCERTAIN", "reason": "<one short sentence>"}}
