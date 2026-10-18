#!/usr/bin/env python3
"""Listening material for the #1819 note cap (docs/experiments/NOTE_CAP.md). No model run.

The model's notes are rebuilt from what the sender actually put on the wire (sender.json `sent` rows: on/off with
planned music time due_ms). Runs with the same seed are independent runs, not pairs: the two arms already diverge
before the cap is first active (left-hand arrival timing changes the context from run to run). So the excerpts that
show the cap are taken from the cap12 arm where the cap blocked a window before its horizon (capped_to_ready > 0):
the capped window with one window before and after, to hear where the next onset was pushed. Dense nocap blocks
(more than CAP notes) are separate general examples, not the same musical position.

  note_cap_listen.py <run root> <lh notes json> <out dir>      (run root: note_cap_run1 with {nocap,cap12}_s{seed}/)
  note_cap_listen.py --page <out dir>                           (index.html from the saved report.json only)
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BEAT, BLOCK = 468.75, 937.5          # 128 BPM, block = 2 beats (scripts/aria_mlx_stream_loop.py)
CAP = 12
SEEDS, SCENARIOS = (1, 2, 3, 4), ("A_normal", "B_crop_early")
CLUSTER_MS = 30                      # onsets this close count as one chord


def planned_notes(sender):
    """Notes the sender sent: [{id, pitch, velocity, on_ms, off_ms, block, open}] by planned music time (due_ms).
    A note whose off was never sent (stopped run) ends at the last due time sent, marked open."""
    on = {r["note_id"]: r for r in sender["sent"] if r["kind"] == "on"}
    off = {r["note_id"]: r["due_ms"] for r in sender["sent"] if r["kind"] == "off"}
    last = max((r["due_ms"] for r in sender["sent"]), default=0)
    out = []
    for nid, r in on.items():
        out.append({"id": nid, "pitch": r["pitch"], "velocity": r["velocity"], "on_ms": r["due_ms"],
                    "off_ms": off.get(nid, max(last, r["due_ms"] + 10)), "block": r["block"], "open": nid not in off})
    return sorted(out, key=lambda n: (n["on_ms"], n["pitch"], n["id"]))


def by_block(notes):
    out = {}
    for n in notes:
        out.setdefault(n["block"], []).append((n["on_ms"], n["pitch"], n["velocity"], None if n["open"] else n["off_ms"]))
    return {k: sorted(v) for k, v in out.items()}


def first_divergence(a, b):
    """First block index whose sent notes differ between two runs (None if identical)."""
    ba, bb = by_block(a), by_block(b)
    return next((k for k in sorted(set(ba) | set(bb)) if ba.get(k) != bb.get(k)), None)


def window_index(on_ms, t0):
    """Block commit window of an onset: [t0 + beat + i * block, t0 + beat + (i + 1) * block) (the cap's grid)."""
    return int((on_ms - (t0 + BEAT)) // BLOCK)


def window_start(i, t0):
    """The cap floor for window i: its start rounded up to the 10 ms onset grid."""
    x = t0 + BEAT + i * BLOCK
    return -(-x // 10) * 10


def excerpt_stats(notes, lo, hi, t0):
    """Notes per window, largest chord (onsets within CLUSTER_MS of the chord's first onset), longest silence and
    onsets placed exactly at a window start (where the cap floor puts a pushed onset) inside [lo, hi)."""
    ns = [n for n in notes if lo <= n["on_ms"] < hi]
    per = {}
    for n in ns:
        w = window_index(n["on_ms"], t0)
        per[w] = per.get(w, 0) + 1
    chord, size, head = 0, 0, None
    for n in ns:
        if head is not None and n["on_ms"] - head <= CLUSTER_MS:
            size += 1
        else:
            head, size = n["on_ms"], 1
        chord = max(chord, size)
    gaps, end = [], lo
    for n in ns:                                           # silence = time not covered by any sounding note
        if n["on_ms"] > end:
            gaps.append(n["on_ms"] - end)
        end = max(end, n["off_ms"])
    if hi > end:
        gaps.append(hi - end)
    at_start = sorted({n["on_ms"] for n in ns if n["on_ms"] == window_start(window_index(n["on_ms"], t0), t0)})
    return {"notes": len(ns), "per_window": {str(k): v for k, v in sorted(per.items())}, "max_window": max(per.values(), default=0),
            "max_chord": chord, "longest_rest_ms": round(max(gaps, default=0.0), 1), "onsets_at_window_start": at_start}


def group_blocks(blocks):
    """Consecutive block numbers merged into (first, last) runs."""
    out = []
    for k in sorted(set(blocks)):
        if out and k == out[-1][1] + 1:
            out[-1][1] = k
        else:
            out.append([k, k])
    return [tuple(x) for x in out]


def lh_notes(lh_path, t0, offset_ms):
    """The fed left hand in music time (feeder: wall origin + offset + note time; music time = t0 + same)."""
    return [{"pitch": p, "velocity": v, "on_ms": t0 + offset_ms + on, "off_ms": t0 + offset_ms + off}
            for p, on, off, v in json.load(open(lh_path))]


def write_pair_midi(rh, lh, lo, hi, path):
    """Two tracks (model right hand, fed left hand) from lo, notes starting in [lo, hi), ends cut at hi + 1 s."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=128)
    for name, ns in (("model", rh), ("left hand", lh)):
        inst = pretty_midi.Instrument(program=0, name=name)
        inst.notes = [pretty_midi.Note(velocity=int(n["velocity"]), pitch=int(n["pitch"]), start=(n["on_ms"] - lo) / 1000,
                                       end=(min(n["off_ms"], hi + 1000) - lo) / 1000)
                      for n in ns if lo <= n["on_ms"] < hi and n["off_ms"] > n["on_ms"]]
        pm.instruments.append(inst)
    pm.write(path)
    back = pretty_midi.PrettyMIDI(path)
    return {"model_notes": len(pm.instruments[0].notes), "read_back": len(back.instruments[0].notes)}


def load_arm(root, arm, seed, sc):
    d = os.path.join(root, f"{arm}_s{seed}", sc)
    snd, prod, feed = (json.load(open(os.path.join(d, f))) for f in ("sender.json", "producer.json", "feeder.json"))
    return {"notes": planned_notes(snd), "t0": snd["origin"][0], "offset": feed["offset_ms"], "prod": prod}


def cut(render, out_dir, name, arm, lh_path, first, last):
    """Windows first - 1 .. last + 1 (block commit windows) as MIDI + audio with stats."""
    t0 = arm["t0"]
    lo, hi = max(t0, t0 + BEAT + (first - 1) * BLOCK), t0 + BEAT + (last + 2) * BLOCK
    w = write_pair_midi(arm["notes"], lh_notes(lh_path, t0, arm["offset"]), lo, hi, os.path.join(out_dir, name))
    render(os.path.join(out_dir, name))
    st = excerpt_stats(arm["notes"], lo, hi, t0)
    nxt = window_start(last + 1, t0)
    sent = {}
    for n in arm["notes"]:
        sent[n["block"]] = sent.get(n["block"], 0) + 1
    committed = {b["block"]: b.get("notes") for b in arm["prod"]["blocks"]}
    rows = {str(k): [sent.get(k, 0), committed.get(k)] for k in range(first - 1, last + 2) if k in committed}
    return {"midi": name, "audio": name[:-4] + ".m4a", "windows": [first - 1, last + 1], "lo_ms": lo, "hi_ms": hi, **w, **st,
            "onset_at_next_window_start": nxt in st["onsets_at_window_start"], "next_window_start_ms": nxt,
            "sent_vs_committed": rows,                      # a stopped run leaves the last block's later notes unsent
            "truncated": any(c is not None and s < c for s, c in rows.values())}


def main() -> None:
    root, lh_path, out_dir = sys.argv[1:4]
    from bebop_fragment_retrieval import render
    os.makedirs(out_dir, exist_ok=True)
    report = {"issue": 1819, "cap": CAP, "runs": [], "cap_excerpts": [], "dense_nocap_examples": [], "full": {}}
    arms = {(a, s, sc): load_arm(root, a, s, sc) for a in ("nocap", "cap12") for s in SEEDS for sc in SCENARIOS}
    for s in SEEDS:                                           # run-to-run divergence record (same seed, independent runs)
        for sc in SCENARIOS:
            n, c = arms[("nocap", s, sc)], arms[("cap12", s, sc)]
            act = [b["block"] for b in c["prod"]["blocks"] if (b.get("capped_samples") or 0) > 0]
            report["runs"].append({"seed": s, "scenario": sc, "first_diverging_block": first_divergence(n["notes"], c["notes"]),
                                   "first_cap_active": act[0] if act else None,
                                   "abort": {"nocap": (n["prod"]["abort"] or {}).get("reason"),
                                             "cap12": (c["prod"]["abort"] or {}).get("reason")}})
    for s in SEEDS:
        for sc in SCENARIOS:
            c = arms[("cap12", s, sc)]
            blocked = [b for b in c["prod"]["blocks"] if (b.get("capped_to_ready") or 0) > 0]
            for first, last in group_blocks(b["block"] for b in blocked):    # the cap blocked a window before its horizon
                name = f"cap12_s{s}_{sc[0]}_b{first}" + (f"-{last}" if last != first else "") + ".mid"
                ex = cut(render, out_dir, name, c, lh_path, first, last)
                ex.update(seed=s, scenario=sc, blocks=[first, last],
                          block_rows=[{k: b.get(k) for k in ("block", "notes", "samples_to_ready", "capped_to_ready", "ready_margin_ms")}
                                      for b in blocked if first <= b["block"] <= last])
                report["cap_excerpts"].append(ex)
                print("cap", s, sc, first, last, ex["per_window"], flush=True)
            n = arms[("nocap", s, sc)]
            for b in n["prod"]["blocks"]:                     # dense nocap blocks: general examples only
                if (b.get("notes") or 0) > CAP:
                    name = f"nocap_s{s}_{sc[0]}_b{b['block']}.mid"
                    ex = cut(render, out_dir, name, n, lh_path, b["block"], b["block"])
                    ex.update(seed=s, scenario=sc, blocks=[b["block"], b["block"]],
                              block_rows=[{k: b.get(k) for k in ("block", "notes", "samples_to_ready", "ready_margin_ms")}])
                    report["dense_nocap_examples"].append(ex)
                    print("dense", s, sc, b["block"], ex["per_window"], flush=True)
    best = max(SEEDS, key=lambda s: sum(1 for b in arms[("cap12", s, "A_normal")]["prod"]["blocks"]
                                        if (b.get("capped_to_ready") or 0) > 0))
    for arm in ("cap12", "nocap"):                            # whole A run of that seed, both arms (independent runs)
        a = arms[(arm, best, "A_normal")]
        t0 = a["t0"]
        hi = max(n["off_ms"] for n in a["notes"]) + 500
        name = f"full_s{best}_A_{arm}.mid"
        w = write_pair_midi(a["notes"], lh_notes(lh_path, t0, a["offset"]), t0, hi, os.path.join(out_dir, name))
        render(os.path.join(out_dir, name))
        report["full"][arm] = {"seed": best, "midi": name, "audio": name[:-4] + ".m4a", **w, **excerpt_stats(a["notes"], t0, hi, t0)}
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


STYLE = """
:root { --bg:#f7f7f5; --card:#fff; --text:#1d1d1f; --muted:#6e6e73; --line:#e3e3e0; --accent:#2f6f4f; --soft:#eef4f0; --bad:#a33a3a; --warn:#9a6a00; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#141414; --card:#1e1e1e; --text:#ececec; --muted:#9a9a9a; --line:#333; --accent:#6fbf8f; --soft:#1d2a22; --bad:#e07070; --warn:#e0b050; color-scheme:dark; } }
:root[data-theme="dark"] { --bg:#141414; --card:#1e1e1e; --text:#ececec; --muted:#9a9a9a; --line:#333; --accent:#6fbf8f; --soft:#1d2a22; --bad:#e07070; --warn:#e0b050; color-scheme:dark; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.6 -apple-system, "Apple SD Gothic Neo", sans-serif; }
main { max-width:1000px; margin:0 auto; padding:24px 16px 80px; }
h1 { font-size:24px; margin:0 0 6px; } h2 { font-size:18px; margin:32px 0 10px; }
.lead { color:var(--muted); margin:0 0 4px; }
.box { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:12px 0; }
.ex { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin:10px 0; }
.ex h3 { font-size:15px; margin:0 0 6px; }
.meta { color:var(--muted); font-size:13px; margin:2px 0 8px; }
.row { display:flex; flex-wrap:wrap; gap:6px 14px; margin:6px 0; font-size:14px; }
.row label { white-space:nowrap; }
.chip { display:inline-block; font-size:12px; padding:1px 8px; border-radius:999px; margin-left:6px; vertical-align:2px; }
.warn { border:1px solid var(--warn); color:var(--warn); } .okc { background:var(--soft); color:var(--accent); }
.tw { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th, td { border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; white-space:nowrap; font-variant-numeric:tabular-nums; }
th { color:var(--muted); font-weight:600; }
audio { width:100%; max-width:440px; }
textarea { width:100%; box-sizing:border-box; font:13px/1.5 ui-monospace, Menlo, monospace; background:var(--card); color:var(--text); border:1px solid var(--line); border-radius:8px; padding:8px; }
button { font:inherit; padding:6px 14px; border-radius:8px; border:1px solid var(--accent); background:var(--soft); color:var(--accent); cursor:pointer; }
ul { margin:6px 0; padding-left:20px; } li { margin:3px 0; }
.muted { color:var(--muted); font-size:13px; }
"""


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _windows(e):
    return " · ".join(f"창 {k}: {v}음" for k, v in e["per_window"].items())


def _card(e, key, options, title, extra=""):
    trunc = '<span class="chip warn">중단 실행: 일부 음 미송출</span>' if e.get("truncated") else ""
    sv = ", ".join(f"블록 {k} {s}/{c}" for k, (s, c) in e.get("sent_vs_committed", {}).items())
    radios = "".join(f'<label><input type="radio" name="{key}" value="{o}"> {o}</label>' for o in options)
    rows = "".join(f"<tr><td>{b['block']}</td><td>{b.get('notes')}</td><td>{b.get('samples_to_ready')}</td>"
                   f"<td>{b.get('capped_to_ready', '-')}</td><td>{b.get('ready_margin_ms')}</td></tr>" for b in e.get("block_rows", []))
    return f"""<div class="ex"><h3>{_esc(title)}{trunc}</h3>
<div class="meta">{_esc(_windows(e))} · 최대 동시음 {e['max_chord']} · 가장 긴 쉼 {e['longest_rest_ms']:.0f} ms{extra}</div>
<audio controls preload="none" src="{_esc(e['audio'])}"></audio>
<div class="tw"><table><tr><th>블록</th><th>확정 음</th><th>지평 전 샘플</th><th>지평 전 하한 활성 샘플</th><th>마감 여유 ms</th></tr>{rows}</table></div>
<div class="muted">송출/확정 음 수: {_esc(sv)}</div>
<div class="row">{radios}</div></div>"""


def page(out_dir):
    """index.html from report.json (Korean listening page; answers are copied as text)."""
    r = json.load(open(os.path.join(out_dir, "report.json")))
    cap_opts = ("자연스러움", "어색함(끊김·몰림)", "모름")
    dense_opts = ("빽빽해서 별로", "괜찮음", "모름")
    full_opts = ("차이 모름", "nocap이 낫다", "cap12가 낫다")
    keys, cards = [], []
    for e in r["cap_excerpts"]:
        k = e["midi"][:-4]
        keys.append(k)
        nxt = " · 다음 창 시작에 바로 음 있음(밀린 onset 후보)" if e["onset_at_next_window_start"] else ""
        cards.append(_card(e, k, cap_opts, f"{k} — seed {e['seed']} {e['scenario'][0]}, 상한 걸린 블록 {e['blocks'][0]}"
                           + (f"~{e['blocks'][1]}" if e['blocks'][1] != e['blocks'][0] else ""), nxt))
    dense = []
    for e in r["dense_nocap_examples"]:
        k = e["midi"][:-4]
        keys.append(k)
        dense.append(_card(e, k, dense_opts, f"{k} — seed {e['seed']} {e['scenario'][0]}, 상한 없이 {e['block_rows'][0]['notes']}음 확정된 블록 {e['blocks'][0]}"))
    full = r["full"]
    fk = f"full_s{full['cap12']['seed']}_A"
    keys.append(fk)
    full_html = "".join(f"""<div class="ex"><h3>{arm} 전체 (seed {f['seed']} A, 독립 실행)</h3>
<div class="meta">음 {f['notes']} · 창 최대 {f['max_window']}음 · 최대 동시음 {f['max_chord']}</div>
<audio controls preload="none" src="{_esc(f['audio'])}"></audio></div>""" for arm, f in full.items())
    full_radios = "".join(f'<label><input type="radio" name="{fk}" value="{o}"> {o}</label>' for o in full_opts)
    runs = "".join(f"<tr><td>{x['seed']}</td><td>{x['scenario'][0]}</td><td>{x['first_diverging_block']}</td>"
                   f"<td>{x['first_cap_active']}</td><td>{x['abort']['nocap']}</td><td>{x['abort']['cap12']}</td></tr>" for x in r["runs"])
    html = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>음 수 상한 청취</title><style>{STYLE}</style></head><body><main>
<h1>음 수 상한 청취 #1819</h1>
<p class="lead">실시간 경로에서 2박(한 블록)에 음을 {r['cap']}개까지만 허용하는 상한을 걸었을 때의 소리입니다. 계산 시간(마감) 때문에 넣은 장치라, 음악이 어떻게 바뀌는지 들어 보시는 것이 목적입니다. 품질 판단 전이며(quality_claimed: false), 위는 모델 오른손 + 입력 왼손을 같이 렌더한 소리입니다.</p>
<div class="box"><b>먼저 알아 둘 점</b><ul>
<li>같은 seed라도 두 팔(상한 없음 nocap, 상한 cap12)은 <b>독립 실행</b>입니다. 상한이 처음 걸리기 전에 이미 갈라져 있어(아래 표), nocap과 cap12를 같은 자리 비교로 들으면 안 됩니다. 차이에는 상한 효과와 실행 간 분기가 섞여 있습니다</li>
<li>그래서 상한 효과는 <b>cap12에서 상한이 실제로 창을 막은 곳</b>(블록의 지평 전 하한 활성 샘플 &gt; 0)으로 들려 드립니다. 막힌 창 앞뒤 한 창씩 포함, 막힌 창은 {r['cap']}음입니다</li>
<li>nocap 예시는 상한 없이 {r['cap']}음을 넘긴 블록의 일반 예시입니다. 같은 음악 위치가 아닙니다</li>
<li><span class="chip warn">중단 실행: 일부 음 미송출</span> 표시는 실행이 capacity_failure로 멈춰 마지막 블록 음이 다 나가지 못한 발췌입니다. 그 창의 밀도는 실제보다 낮게 들립니다(안전 종료 결함은 별도 수정 중)</li></ul></div>

<h2>1. 상한이 걸린 곳 (cap12)</h2>
<p class="muted">질문: 상한이 걸린 자리가 자연스럽게 들리나요, 음이 끊기거나 다음 박으로 몰려 어색한가요?</p>
{''.join(cards)}

<h2>2. 상한 없이 빽빽했던 블록 (nocap, 일반 예시)</h2>
<p class="muted">질문: 이런 빽빽한 블록이 별로인가요, 괜찮은가요? (상한이 필요한지 판단용)</p>
{''.join(dense)}

<h2>3. 전체 길이 (seed {full['cap12']['seed']} A, 독립 실행 두 개)</h2>
<p class="muted">cap12에서 상한이 가장 많이 걸린 실행과, 같은 seed의 nocap 실행입니다. 위치별 비교가 아니라 전체 인상 비교입니다.</p>
{full_html}
<div class="row">{full_radios}</div>

<h2>답 보내기</h2>
<p class="muted">고른 뒤 버튼을 누르고, 아래 글을 복사해 붙여 주세요.</p>
<textarea id="memo" rows="3" placeholder="메모(선택)"></textarea>
<p><button onclick="build()">답 글 만들기</button> <button onclick="copyOut()">복사</button></p>
<textarea id="out" rows="14" readonly></textarea>

<h2>참고: 실행 간 분기</h2>
<div class="tw"><table><tr><th>seed</th><th>시나리오</th><th>송출 음이 처음 다른 블록</th><th>상한 첫 활성 블록</th><th>nocap 중단</th><th>cap12 중단</th></tr>{runs}</table></div>
<p class="muted">원자료 note_cap_run1/, 보고 report.json, 스크립트 scripts/note_cap_listen.py</p>
<script>
const KEYS = {json.dumps(keys, ensure_ascii=False)};
function build() {{
  const lines = ["[음 수 상한 청취 #1819]"];
  let n = 0;
  for (const k of KEYS) {{
    const c = document.querySelector('input[name="' + k + '"]:checked');
    if (c) n++;
    lines.push(k + ": " + (c ? c.value : "-"));
  }}
  const memo = document.getElementById("memo").value.trim();
  if (memo) lines.push("메모: " + memo);
  lines.push("응답 " + n + "/" + KEYS.length);
  document.getElementById("out").value = lines.join("\\n");
}}
function copyOut() {{
  const t = document.getElementById("out");
  if (!t.value) build();
  t.select();
  try {{ navigator.clipboard.writeText(t.value); }} catch (e) {{ document.execCommand("copy"); }}
}}
</script>
</main></body></html>
"""
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(html)


if __name__ == "__main__":
    if sys.argv[1] == "--page":
        page(sys.argv[2])
    else:
        main()
        page(sys.argv[3])
