# Split vs Native Speculative Timing (2026-04-03)

## Setup

- machine: Apple M4 (Metal backend)
- draft model: `qwen2.5-0.5b-instruct-q4_k_m.gguf`
- verify/target model: `qwen2.5-1.5b-instruct-q4_k_m.gguf`
- prompt: `Explain speculative decoding briefly.`
- `n_max=4`, `max_output_tokens=24`, `ctx=512`

## Compared implementations

1. split-native-full
   - `llama-spec-split-draft` + `llama-spec-split-verify`
   - file-bus exchange via `state/proposal/decision`
2. native in-process baseline
   - `llama-speculative-simple`
   - same model pair and greedy sampling

## Instrumentation added

- split draft:
  - sync/rollback alignment time
  - tail-logits refresh time
  - draft generation time
  - `decision_write -> next_draft_seen` communication latency
- split verify:
  - `proposal_write -> verify_seen` communication latency
  - verify decode time
  - verify sample-and-accept time
  - KV rollback time
- native speculative-simple:
  - draft stage time
  - target decode stage time
  - sample-and-accept stage time
  - post stage (`llama_memory_seq_rm`)

## One-run summary (7 rounds)

- split:
  - draft generation avg: `30.51 ms`
  - draft sync+tail prep avg: `1.95 + 11.94 ms`
  - proposal->verify communication avg: `82.52 ms`
  - decision->draft communication avg: `160.48 ms`
  - verify decode avg: `6.22 ms` (warm rounds near `0.64 ms`)
  - verify sample avg: `47.19 ms`
  - verify rollback avg: `0.0024 ms`
  - reject rounds: `4`
- native speculative-simple:
  - draft avg: `36.27 ms`
  - decode avg: `0.87 ms`
  - sample avg: `47.15 ms`
  - post avg: `0.0019 ms`
  - reject rounds: `4`

## Warm-round comparison (excluding round 0)

- split proposal->verify comm avg: `74.54 ms`
- split decision->draft comm avg: `160.48 ms`
- split draft compute avg: `25.26 ms`
- split verify decode avg: `0.635 ms`
- split verify sample avg: `47.52 ms`

- native draft compute avg: `36.27 ms`
- native decode avg: `0.869 ms`
- native sample avg: `47.15 ms`
- native total loop avg: `84.30 ms`

## Conclusion

- No severe discrepancy in core speculative logic stages:
  - verify sample and rollback are aligned in scale between split and native.
- Main performance gap comes from process exchange overhead in split mode:
  - especially `decision -> next draft` latency.
- To narrow the gap, reduce polling/file I/O overhead (event signaling or RPC stream/session reuse).
