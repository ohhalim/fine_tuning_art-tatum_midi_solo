#!/usr/bin/env python3
"""Blinded listening pairs for the density tilt (issue #1844, docs/experiments/DENSITY_TILT.md 7절). No model run.

Paired runs (deterministic left hand) are identical until the tilt first acts, so the same windows of both arms can be
compared: for each seed, the first A-scenario block where the tilt acted before the horizon, with one window before and
after. Plus longer excerpts (the whole A run with the most tilt activity). A/B order is random (seed 1844); the key is
in key.json and the report. The page has working inputs (radio buttons, memo, a button that builds and copies the
answer) — not a read-only box.

  density_tilt_listen.py <test run dir> <lh notes json> <out dir> [--max-pairs 8]
"""
from __future__ import annotations

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from note_cap_listen import BEAT, BLOCK, excerpt_stats, lh_notes, planned_notes, write_pair_midi  # noqa: E402

SEEDS = tuple(range(1, 17))


def load(run_dir, arm, seed):
    d = os.path.join(run_dir, f"test_{arm}_s{seed}", "A_normal")
    snd, prod = json.load(open(os.path.join(d, "sender.json"))), json.load(open(os.path.join(d, "producer.json")))
    return {"notes": planned_notes(snd), "t0": snd["origin"][0], "offset": (prod.get("lh_source") or {}).get("offset_ms", 937.5),
            "prod": prod}


def excerpt(render, out_dir, name, arm, lh_path, first, last):
    t0 = arm["t0"]
    lo, hi = max(t0, t0 + BEAT + (first - 1) * BLOCK), t0 + BEAT + (last + 2) * BLOCK
    w = write_pair_midi(arm["notes"], lh_notes(lh_path, t0, arm["offset"]), lo, hi, os.path.join(out_dir, name))
    render(os.path.join(out_dir, name))
    st = excerpt_stats(arm["notes"], lo, hi, t0)
    return {"audio": name[:-4] + ".m4a", "midi": name, "windows": [first - 1, last + 1], **w,
            "per_window": st["per_window"], "max_window": st["max_window"], "max_chord": st["max_chord"], "longest_rest_ms": st["longest_rest_ms"]}


def page(out_dir, items):
    rows = []
    for i, it in enumerate(items, 1):
        rows.append(f"""<div class='c'><b>{i}. {it['label']}</b>
<p>A<br><audio controls preload='none' src='{it['A']['audio']}'></audio></p>
<p>B<br><audio controls preload='none' src='{it['B']['audio']}'></audio></p>
<label><input type='radio' name='q{i}' value='A'> A가 낫다</label>
<label><input type='radio' name='q{i}' value='B'> B가 낫다</label>
<label><input type='radio' name='q{i}' value='차이 모름'> 차이 모름</label>
<label><input type='radio' name='q{i}' value='둘 다 어색함'> 둘 다 어색함</label><br>
<input type='text' id='m{i}' placeholder='메모(선택)' style='width:100%;margin-top:4px'></div>""")
    n = len(items)
    html = f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>밀도 기울기 청취</title><style>:root{{--bg:#fff;--fg:#1d1d1f;--mu:#6e6e73;--card:#f5f5f7}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--mu:#9a9aa0;--card:#1c1c1e}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.5 -apple-system,sans-serif;margin:0 auto;max-width:760px;padding:16px}}
.c{{background:var(--card);border-radius:12px;padding:12px 16px;margin:12px 0}}audio{{width:100%}}textarea{{width:100%;height:120px}}
label{{margin-right:10px;white-space:nowrap}}</style></head><body>
<h1>밀도 기울기 청취 (#1844)</h1>
<p>음이 몰리는 구간에서 다음 음을 조금 늦게 치도록 기울인 쪽과 기울기 없는 쪽. 같은 왼손·같은 seed라 기울기가 처음 작동하기 전까지는 같은 연주.
A·B 중 어느 쪽이 기울기인지는 숨김. 품질 주장 없음.</p>
{''.join(rows)}
<div class='c'><b>답</b><p><label>전체 메모<br><textarea id='all'></textarea></label></p>
<button id='go' style='font-size:16px;padding:8px 16px'>답 만들고 복사</button>
<textarea id='out' style='margin-top:8px' placeholder='버튼을 누르면 여기에 답이 생기고 복사됩니다. 이 내용을 채팅에 붙여 주세요.'></textarea></div>
<script>
document.getElementById('go').onclick=function(){{
 var L=['[밀도 기울기 청취 #1844]'];
 for(var i=1;i<={n};i++){{var r=document.querySelector('input[name=q'+i+']:checked');
  var m=document.getElementById('m'+i).value.trim();
  L.push(i+'. '+(r?r.value:'미응답')+(m?' — '+m:''));}}
 var a=document.getElementById('all').value.trim(); if(a)L.push('메모: '+a);
 var o=document.getElementById('out'); o.value=L.join('\\n'); o.select();
 try{{navigator.clipboard.writeText(o.value);}}catch(e){{document.execCommand('copy');}}
}};
</script></body></html>"""
    open(os.path.join(out_dir, "index.html"), "w").write(html)


def main() -> None:
    run_dir, lh_path, out_dir = sys.argv[1:4]
    max_pairs = int(sys.argv[sys.argv.index("--max-pairs") + 1]) if "--max-pairs" in sys.argv else 8
    from bebop_fragment_retrieval import render
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(1844)
    items, key = [], []
    best, best_n = None, -1
    for s in SEEDS:
        off, tilt = load(run_dir, "off", s), load(run_dir, "tilt", s)
        act = [b["block"] for b in tilt["prod"]["blocks"] if b.get("tilted_to_ready")]
        if len(act) > best_n:
            best, best_n = s, len(act)
        if not act or len(items) >= max_pairs:
            continue
        b = act[0]
        sides = [("off", off), ("tilt", tilt)]
        if rng.random() < 0.5:
            sides.reverse()
        it = {"label": f"짧은 발췌 seed {s} 블록 {b} 앞뒤", "seed": s, "block": b}
        for lab, (arm, data) in zip("AB", sides):
            it[lab] = excerpt(render, out_dir, f"p{s}_b{b}_{lab}.mid", data, lh_path, b, b)
            it[lab]["arm"] = arm
        items.append(it)
        key.append({"item": len(items), "seed": s, "block": b, "A": sides[0][0], "B": sides[1][0]})
    if best is not None:
        off, tilt = load(run_dir, "off", best), load(run_dir, "tilt", best)
        last = max(b["block"] for b in tilt["prod"]["blocks"])
        sides = [("off", off), ("tilt", tilt)]
        if rng.random() < 0.5:
            sides.reverse()
        it = {"label": f"긴 발췌 seed {best} 전체", "seed": best, "block": None}
        for lab, (arm, data) in zip("AB", sides):
            it[lab] = excerpt(render, out_dir, f"full_s{best}_{lab}.mid", data, lh_path, 1, last)
            it[lab]["arm"] = arm
        items.append(it)
        key.append({"item": len(items), "seed": best, "block": "full", "A": sides[0][0], "B": sides[1][0]})
    page(out_dir, items)
    json.dump(key, open(os.path.join(out_dir, "key.json"), "w"), ensure_ascii=False, indent=1)
    json.dump({"issue": 1844, "items": items, "key": key}, open(os.path.join(out_dir, "report.json"), "w"), ensure_ascii=False, indent=1)
    print(len(items), "items", os.path.join(out_dir, "index.html"))


if __name__ == "__main__":
    main()
