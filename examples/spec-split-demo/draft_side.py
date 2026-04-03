#!/usr/bin/env python3
import json
import os
import random
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


N_MAX = 4
MISMATCH_PROB = 0.35
POLL_SEC = 0.2
DRAFT_ENDPOINT = os.getenv("DRAFT_ENDPOINT", "http://127.0.0.1:8091")


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_document_tail(max_chars: int = 120) -> str:
    if not DOCUMENT.exists():
        return ""
    text = DOCUMENT.read_text(encoding="utf-8").strip()
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]


def read_document_text() -> str:
    if not DOCUMENT.exists():
        return ""
    text = DOCUMENT.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    return "\n".join(lines).strip()


def split_tokens(text: str):
    return [tok for tok in text.replace("\n", " ").split(" ") if tok]


def make_draft(target_tokens, accepted_pos: int, n_max: int, rng: random.Random):
    start = accepted_pos + 1
    end = min(start + n_max, len(target_tokens))
    draft = target_tokens[start:end]
    if not draft:
        return draft

    if rng.random() < MISMATCH_PROB:
        i = rng.randrange(len(draft))
        # deterministic "wrong" token for easier debugging
        draft[i] = "<WRONG>"
    return draft


def model_generate_draft(prompt: str, generated_text: str, n_max: int):
    req_obj = {
        "prompt": (
            f"{prompt}\n\nCurrent output:\n{generated_text}\n\n"
            f"Continue naturally with at most {n_max} tokens. Return plain text only."
        ),
        "n_predict": n_max,
        "temperature": 0.2,
        "top_k": 20,
        "top_p": 0.9,
    }
    req = urllib.request.Request(
        f"{DRAFT_ENDPOINT}/completion",
        data=json.dumps(req_obj).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return split_tokens(payload.get("content", ""))


def main() -> None:
    rng = random.Random(1234)
    print("[draft] started")
    last_round_logged = -1

    while True:
        state = read_json(STATE)
        config = read_json(CONFIG)
        if state is None:
            print("[draft] missing state.json, run init_demo.py first")
            return
        if config is None:
            print("[draft] missing config.json, run init_demo.py first")
            return

        if state["done"]:
            print("[draft] done=true, exit")
            return

        mode = state.get("mode", "toy")
        round_id = state["round"]
        accepted_pos = state["accepted_pos"]
        n_max = int(config.get("n_max", N_MAX))
        id_last = accepted_pos

        if round_id != last_round_logged:
            doc_tail = read_document_tail()
            print(f"[draft] observe doc before round {round_id}: {doc_tail!r}")
            last_round_logged = round_id

        cur_proposal = read_json(PROPOSAL)
        if cur_proposal is not None and cur_proposal.get("round") == round_id:
            # wait for verifier decision
            decision = read_json(DECISION)
            if decision is not None and decision.get("round") == round_id:
                doc_tail = read_document_tail()
                print(
                    f"[draft] round {round_id} decision: "
                    f"accepted_draft={decision['accepted_draft']} "
                    f"result={decision['result_tokens']} "
                    f"doc_tail={doc_tail!r}"
                )
            time.sleep(POLL_SEC)
            continue

        if mode == "toy":
            target_tokens = state["target_tokens"]
            id_last = target_tokens[accepted_pos]
            draft_tokens = make_draft(target_tokens, accepted_pos, n_max, rng)
        else:
            prompt = str(config.get("prompt", ""))
            generated_text = read_document_text()
            try:
                draft_tokens = model_generate_draft(prompt, generated_text, n_max)
            except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as e:
                print(f"[draft] model call failed in round {round_id}: {e}")
                time.sleep(POLL_SEC)
                continue

        proposal = {
            "round": round_id,
            "accepted_pos": accepted_pos,
            "id_last": id_last,
            "draft_tokens": draft_tokens,
            "n_max": n_max,
        }
        write_json(PROPOSAL, proposal)
        print(
            f"[draft] round {round_id} proposal: "
            f"id_last={id_last} draft={draft_tokens}"
        )
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
