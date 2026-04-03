# Spec Split Demo (File Bus)

This demo shows how to separate speculative drafting and verification into two independent processes.

- `draft_side.py` only proposes draft tokens and writes a proposal file.
- `verify_side.py` reads the proposal, verifies it against a target stream, appends accepted output to a document, and writes a decision file.

Communication is done through shared files:

- `shared/state.json`
- `shared/proposal.json`
- `shared/decision.json`
- `shared/document.md`

## 1) Initialize

```bash
cd /Users/doffin_azure/Code/Project/llama.cpp/examples/spec-split-demo
./setup_env.sh
source .venv/bin/activate
python3 init_demo.py
```

## 2) Run verifier (terminal A)

```bash
python3 verify_side.py
```

## 3) Run drafter (terminal B)

```bash
python3 draft_side.py
```

The two processes will advance round by round until `done=true`.

## Notes

- This is a protocol demo, not real model inference.
- The verifier simulates `sample_and_accept_n` behavior:
  - accept matching draft prefix
  - stop at first mismatch and emit target token
  - if full draft matches, emit one extra target token
- `document.md` acts as the shared "manuscript" written by verifier and readable by drafter/others.

## One-command run

```bash
./setup_env.sh
./run_demo.sh
tail -f shared/verify.log
tail -f shared/draft.log
./clean_demo.sh
```

## Real-model mode (minimal extension)

This demo now supports a `model` mode that keeps the same file-bus protocol while calling real models via `llama-server`.

- draft side calls `DRAFT_ENDPOINT/completion` to propose `n_max` tokens
- verify side calls `VERIFY_ENDPOINT/completion` to sample `n_max + 1` tokens
- verifier still applies prefix-accept logic and writes accepted text into `document.md`

### Quick start

```bash
cd /Users/doffin_azure/Code/Project/llama.cpp/examples/spec-split-demo
./setup_env.sh
./run_model_demo.sh
tail -f shared/verify.log
tail -f shared/draft.log
```

### Useful overrides

```bash
DRAFT_MODEL=/abs/path/draft.gguf \
VERIFY_MODEL=/abs/path/verify.gguf \
PROMPT="Write 2 short lines about speculative decoding." \
N_MAX=4 MAX_OUTPUT_TOKENS=64 CTX_SIZE=512 \
./run_model_demo.sh
```

### Cross-machine split

When draft and verify are on different machines, run two `llama-server` endpoints independently and point the processes to remote addresses:

```bash
DRAFT_ENDPOINT=http://draft-host:8091 \
VERIFY_ENDPOINT=http://verify-host:8092 \
python draft_side.py

DRAFT_ENDPOINT=http://draft-host:8091 \
VERIFY_ENDPOINT=http://verify-host:8092 \
python verify_side.py
```
