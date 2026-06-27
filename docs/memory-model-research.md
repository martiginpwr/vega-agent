# Memory Model Research

This note records the first local model strategy for Vega's automatic memory pipeline.

## Requirement

Vega needs a local background model that can decide whether a conversation turn should become memory. This is not a simple label-only task. The model should return structured output:

- whether to store memory
- memory type
- concise memory text
- confidence
- importance
- rationale
- conflict or duplicate hints
- source message IDs

Because of that, the first memory model should be a small instruction-tuned language model rather than a pure keyword heuristic or classic classifier.

## Candidate Models

### Qwen2.5-0.5B-Instruct

Hugging Face: <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct>

Why it is attractive:

- About 0.5B parameters.
- Instruction tuned.
- Model card highlights improvements in instruction following and structured JSON output.
- Context length is 32,768 tokens.
- Official GGUF quantization exists: <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF>
- Apache 2.0 license.

Recommendation:

- Use this as the first memory classifier/extractor.
- Run it with low temperature and strict JSON schema prompting.
- Keep input small: latest turn plus compact conversation/session metadata plus top relevant existing memories.

### Qwen3-0.6B

Hugging Face: <https://huggingface.co/Qwen/Qwen3-0.6B>

Why it is attractive:

- About 0.6B parameters.
- Context length is 32,768 tokens.
- Apache 2.0 license.
- Good family match if the main model is Qwen.

Concern:

- Qwen3 has thinking mode behavior. For background classification we want fast, deterministic, compact JSON, so the runtime would need to disable thinking or enforce `/no_think`.

Recommendation:

- Evaluate after Qwen2.5-0.5B-Instruct.
- It may be useful if it classifies better, but it is not the first default because thinking-mode handling adds complexity.

### SmolLM2-1.7B-Instruct

Hugging Face: <https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct>

Why it is attractive:

- Compact 1.7B-class instruction model.
- Model card describes on-device use and instruction following.
- Function-calling examples suggest it may handle structured tasks better than smaller models.
- GGUF quantizations exist, including <https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF>.

Concern:

- Larger than needed for the first always-on memory pass.

Recommendation:

- Use as a verifier or quality upgrade candidate if Qwen2.5-0.5B is too noisy.

### Phi-3.5-mini-instruct

Hugging Face: <https://huggingface.co/microsoft/Phi-3.5-mini-instruct>

Why it is attractive:

- Strong small instruct model.
- Good candidate for high-quality verification.

Concern:

- Around 3.8B parameters, which is too heavy for every background memory turn on a 12 GB VRAM / 16 GB RAM machine when a larger chat model may already be loaded.

Recommendation:

- Do not use as the default memory classifier.
- Consider only for occasional offline audits or high-value verification.

### Classic Classifiers And Embeddings

Useful supporting models:

- `sentence-transformers/all-MiniLM-L6-v2`: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
- `BAAI/bge-small-en-v1.5`: <https://huggingface.co/BAAI/bge-small-en-v1.5>
- `MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33`: <https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33>
- `facebook/bart-large-mnli`: <https://huggingface.co/facebook/bart-large-mnli>

Recommendation:

- Use embeddings to narrow existing memories for dedupe/conflict checking.
- Do not use zero-shot classifiers as the first memory extractor because they do not naturally produce the memory content and rationale Vega needs.

## Proposed Memory Pipeline

```text
new turn saved to SQLite
  -> build memory review packet
  -> Qwen2.5-0.5B-Instruct classifier/extractor
  -> validate JSON against schema
  -> embed candidate memory
  -> retrieve top similar active memories
  -> verifier decision for medium-confidence or conflict-prone candidates
  -> write active, suggested, rejected, or no-op record
```

## Verifier Strategy

Do not stuff all memory into the verifier. Use retrieval first:

1. Embed the candidate memory.
2. Retrieve the top 5-10 similar existing memories by vector similarity.
3. Include only those memories plus compact metadata in the verifier prompt.
4. Ask for one of:
   - `save_new`
   - `merge_with_existing`
   - `update_existing`
   - `reject_duplicate`
   - `reject_low_value`
   - `mark_conflict`

The verifier can be the same small model at first. A second model should be introduced only if evaluation shows meaningful improvement.

## First Default

Use:

- Classifier/extractor: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
- Embedding/dedupe: existing local Qwen embedding model or `bge-small-en-v1.5`
- Verifier: same classifier model for medium-confidence/conflict cases

This keeps the system local, fast, and realistic on the target machine.
