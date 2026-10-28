"""Optional outside reference: zero-shot Qwen2.5-VL-7B-Instruct asked "Is this image AI generated?".

Not part of any hypothesis test. The score is log P("Yes") - log P("No") for the first answer
token, so it can be evaluated with the same AUROC code. Needs a 24 GB GPU (A100/L4) in bf16.

    python scripts/zero_shot_mllm.py --config configs/default.yaml --limit 2000
Output: <run_dir>/scores/qwen25vl_zeroshot/clean.csv
"""

from __future__ import annotations

import pandas as pd

from _bootstrap import config, parser
from prism3d.pipeline import load_manifest, processed_path, run_dir

PROMPT = "Is this image AI generated? Answer with Yes or No only."

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    cfg = config(a)
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    man = load_manifest(cfg)
    man = man[man.split == "test"]
    if a.limit:
        man = man.sample(min(a.limit, len(man)), random_state=0)
    rows = []
    for r in man.itertuples():
        img = Image.open(processed_path(cfg, r.image_id, "clean")).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        rows.append({"image_id": r.image_id, "logit": float(lp[yes_id] - lp[no_id])})
    out = run_dir(cfg) / "scores" / "qwen25vl_zeroshot"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "clean.csv", index=False)
    print(f"Wrote {len(rows)} scores to {out / 'clean.csv'}")
