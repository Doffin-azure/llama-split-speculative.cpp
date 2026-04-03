#!/usr/bin/env python3
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SHARED = ROOT / "shared"
STATE = SHARED / "state.json"
PROPOSAL = SHARED / "proposal.json"
DECISION = SHARED / "decision.json"
DOCUMENT = SHARED / "document.md"
CONFIG = SHARED / "config.json"


POLL_SEC = 0.2
VERIFY_ENDPOINT = os.getenv("VERIFY_ENDPOINT", "http://127.0.0.1:8092")


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def verify_like_sample_and_accept_n(target_tokens, accepted_pos, draft_tokens):
    # Simulate common_sampler_sample_and_accept_n:
    # - accept matching draft prefix
    # - break on first mismatch and emit target token at mismatch position
    # - if full draft matched, emit one extra target token
    result = []
    i = 0
    while i < len(draft_tokens):
        pos = accepted_pos + i + 1
        if pos >= len(target_tokens):
            break
        sampled = target_tokens[pos]
        result.append(sampled)
        if draft_tokens[i] != sampled:
            break
        i += 1

    if i == len(draft_tokens):
        pos = accepted_pos + i + 1
        if pos < len(target_tokens):
            result.append(target_tokens[pos])

    accepted_draft = max(0, len(result) - 1)
    return result, accepted_draft


def verify_like_prefix(draft_tokens, sampled_tokens):
    if not sampled_tokens:
        return [], 0

    i = 0
    while i < len(draft_tokens) and i < len(sampled_tokens):
        if draft_tokens[i] != sampled_tokens[i]:
            break
        i += 1

    if i < len(sampled_tokens) and (i >= len(draft_tokens) or draft_tokens[i] != sampled_tokens[i]):
        result = sampled_tokens[: i + 1]
    else:
        extra = 1 if len(sampled_tokens) > i else 0
        result = sampled_tokens[: i + extra]

    accepted_draft = i
    return result, accepted_draft


def read_document_text() -> str:
    if not DOCUMENT.exists():
        return ""
    text = DOCUMENT.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    return "\n".join(lines).strip()


def split_tokens(text: str):
    return [tok for tok in text.replace("\n", " ").split(" ") if tok]


def model_generate_sample(prompt: str, generated_text: str, n_predict: int):
    req_obj = {
        "prompt": (
            f"{prompt}\n\nCurrent output:\n{generated_text}\n\n"
            f"Continue naturally with at most {n_predict} tokens. Return plain text only."
        ),
        "n_predict": n_predict,
        "temperature": 0.2,
        "top_k": 20,
        "top_p": 0.9,
    }
    req = urllib.request.Request(
        f"{VERIFY_ENDPOINT}/completion",
        data=json.dumps(req_obj).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return split_tokens(payload.get("content", ""))


def append_document(tokens):
    if not tokens:
        return
    text = " ".join(tokens)
    with DOCUMENT.open("a", encoding="utf-8") as f:
        f.write(text + " ")


def main() -> None:
    print("[verify] started")

    while True:
        state = read_json(STATE)
        config = read_json(CONFIG)
        if state is None:
            print("[verify] missing state.json, run init_demo.py first")
            return
        if config is None:
            print("[verify] missing config.json, run init_demo.py first")
            return

        if state["done"]:
            print("[verify] done=true, exit")
            return

        proposal = read_json(PROPOSAL)
        if proposal is None:
            time.sleep(POLL_SEC)
            continue

        if proposal["round"] != state["round"]:
            time.sleep(POLL_SEC)
            continue

        if proposal["accepted_pos"] != state["accepted_pos"]:
            print("[verify] stale proposal ignored")
            time.sleep(POLL_SEC)
            continue

        mode = state.get("mode", "toy")
        if mode == "toy":
            target_tokens = state["target_tokens"]
            result_tokens, accepted_draft = verify_like_sample_and_accept_n(
                target_tokens=target_tokens,
                accepted_pos=state["accepted_pos"],
                draft_tokens=proposal["draft_tokens"],
            )
            done = state["accepted_pos"] + len(result_tokens) >= len(target_tokens) - 1
        else:
            prompt = str(config.get("prompt", ""))
            generated_text = read_document_text()
            n_max = int(config.get("n_max", proposal.get("n_max", 4)))
            max_output_tokens = int(config.get("max_output_tokens", 64))
            try:
                sampled_tokens = model_generate_sample(
                    prompt=prompt, generated_text=generated_text, n_predict=n_max + 1
                )
            except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as e:
                print(f"[verify] model call failed in round {state['round']}: {e}")
                time.sleep(POLL_SEC)
                continue
            result_tokens, accepted_draft = verify_like_prefix(
                draft_tokens=proposal["draft_tokens"], sampled_tokens=sampled_tokens
            )
            done = state["accepted_pos"] + len(result_tokens) >= max_output_tokens

        new_accepted_pos = state["accepted_pos"] + len(result_tokens)

        append_document(result_tokens)

        decision = {
            "round": state["round"],
            "accepted_draft": accepted_draft,
            "result_tokens": result_tokens,
            "new_accepted_pos": new_accepted_pos,
            "done": done,
        }
        write_json(DECISION, decision)

        next_state = dict(state)
        next_state["accepted_pos"] = new_accepted_pos
        next_state["round"] = state["round"] + 1
        next_state["done"] = done
        write_json(STATE, next_state)

        try:
            PROPOSAL.unlink()
        except FileNotFoundError:
            pass

        print(
            f"[verify] round {state['round']} verified: "
            f"accepted_draft={accepted_draft}, emitted={result_tokens}"
        )

        if done:
            print("[verify] target stream fully emitted")
            return

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
