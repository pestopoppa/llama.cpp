#!/usr/bin/env python3
"""Compare matched greedy plain/MTP response token ids without inferring state equality."""
import argparse, json, sys
from pathlib import Path

def measured(root: Path):
    out = []
    for response_path in sorted(root.glob("response-measure-*.json")):
        suffix = response_path.name.removeprefix("response-")
        request_path = root / f"request-{suffix}"
        if not request_path.is_file():
            raise RuntimeError(f"{root}: missing saved payload {request_path.name}")
        request = json.loads(request_path.read_text())
        if request.get("n_predict") != 512:
            raise RuntimeError(f"{request_path}: expected n_predict=512")
        doc = json.loads(response_path.read_text())
        tokens = doc.get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError(f"{response_path}: return_tokens response missing token list")
        if len(tokens) != 512:
            raise RuntimeError(f"{response_path}: expected 512 returned tokens, got {len(tokens)}")
        if any(isinstance(token, bool) or not isinstance(token, int) for token in tokens):
            raise RuntimeError(f"{response_path}: token list contains non-integer token ids")
        out.append((response_path.name, request, tokens))
    if len(out) != 5:
        raise RuntimeError(f"{root}: expected exactly five measured responses")
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", type=Path, required=True)
    ap.add_argument("--mtp", type=Path, required=True)
    args = ap.parse_args()
    plain, mtp = measured(args.plain), measured(args.mtp)
    failed = []
    for (pn, pr, pt), (mn, mr, mt) in zip(plain, mtp):
        if pn != mn or pr != mr or pt != mt:
            first = next((i for i, pair in enumerate(zip(pt, mt)) if pair[0] != pair[1]), min(len(pt), len(mt)))
            failed.append({"plain": pn, "mtp": mn, "first_difference": first,
                           "plain_len": len(pt), "mtp_len": len(mt),
                           "request_payload_equal": pr == mr})
    result = {"schema": "epyc.glm53.matched_greedy_mtp_plain.v1",
              "scope": "same-request end-to-end greedy token identity; not raw state equality",
              "pairs": 5, "verdict": "PASS" if not failed else "FAIL", "differences": failed}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failed else 3

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        raise SystemExit(2)
