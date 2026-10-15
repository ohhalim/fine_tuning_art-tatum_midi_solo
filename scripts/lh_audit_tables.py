#!/usr/bin/env python3
"""Markdown result tables for the left-hand audits, generated from the saved report only (#1750, #1754).

  lh_audit_tables.py <report json> [<#1750 ideal report json, for the comparison column>]

The "ideal" report (#1750) gives the tables pasted in docs/experiments/CAUSAL_LH_AUDIT.md; the "runtime_block" report
(#1754) gives the tables for docs/experiments/BLOCK_SNAPSHOT_LH_AUDIT.md.
"""
from __future__ import annotations

import json
import sys

R = ("g3", "top50", "grid300")
CONDS = ("sounding", "recent")
PHASES = ("block_p0", "block_p1", "block_p2", "block_p3")


def pct(x):
    return f"{x*100:.1f}%" if x is not None else "–"


def ideal_tables(r):
    m, f = r["strata"]["main"], r["strata"]["afable"]
    out = ["**검사 (주대상 189곡, 전체 타깃)**\n",
           "| 규칙 | 타깃 | 온라인 K1/K2/K3 | replay 불일치 | prefix·미래 교체 표본 | prefix 불일치 | off 노출 | 미래 교체 불일치 | 구성 가능 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for k in R:
        v = m["verdicts"][k]
        L = v["online_leaks"]
        out.append(f"| `{k}` | {m['rules'][k]['targets']:,} | {L['K1']}/{L['K2']}/{L['K3']} | {v['replay_mismatch']} | {v['samples']:,} | "
                   f"{v['prefix_mismatch']} | {v['prefix_finite_off_exposed']} | {v['suffix_mismatch']} | {'예' if v['construction_ok'] else '아니오'} |")
    out += ["\n**비어 있지 않은 비율 (주대상, 결정 시각 = onset − L)**\n",
            "| 규칙 | 조건 | L = 0 | L = 1박 (판정) | 1박 원래 개수 | 곡별 최소 · p10 · 중앙 · p90 (1박) | 0인 곡 | 판정 |",
            "|---|---|---|---|---|---|---|---|"]
    for k in R:
        for c in CONDS:
            a, b = m["rules"][k]["online"]["lead0"][c], m["rules"][k]["online"]["lead1beat"][c]
            ps, vc = b["per_song_nonempty_ratio"], m["verdicts"][k]["conditions"][c]
            out.append(f"| `{k}` | {c} | {pct(a['nonempty_ratio'])} | {pct(b['nonempty_ratio'])} | {b['nonempty']:,}/{b['targets']:,} | "
                       f"{pct(ps['min'])} · {pct(ps['p10'])} · {pct(ps['median'])} · {pct(ps['p90'])} | {b['songs_with_zero_nonempty']} | {vc['verdict']} |")
    out += ["\n**오프라인 지름길이 쓰는 미래 정보 (주대상)**\n",
            "| 규칙 | 기준 | 비어 있지 않음 | 조건 항목 중 미래 onset (K2) | 역할 미확정 (K3) | 누출 있는 타깃 | 미래 떼기로 뺀 음이 있는 타깃 (K4) | 온라인 sounding과 Jaccard 중앙 |",
            "|---|---|---|---|---|---|---|---|"]
    for k in R:
        for ln, nm in (("lead0", "L = 0"), ("lead1beat", "L = 1박")):
            o = m["rules"][k]["offline"][ln]
            out.append(f"| `{k}` | {nm} | {pct(o['nonempty_ratio'])} | {pct(o['entries_future_onset_share'])} | {pct(o['entries_role_unknown_share'])} | "
                       f"{pct(o['targets_with_leak_share'])} | {pct(o['targets_with_K4_share'])} | {o['jaccard_vs_sounding']['median']} |")
    out += ["\n**남은 타깃, PC 다양성, 시간 (주대상, 1박)**\n",
            "| 규칙 | 타깃 / 전체 음 | 제외 | 역할 확정 지연 중앙 · 최대 (ms) | 조건 | 평균 크기 | PC 집합 크기 중앙 | 서로 다른 PC 집합 | 엔트로피 (bit) | 상위 5 비중 | 조건 나이 d − on 중앙 · p90 (ms) | d < 0 타깃 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in R:
        t = m["rules"][k]
        ex = sum(v for kk, v in t["role_counts"].items() if kk.startswith("excluded"))
        rd = t["time_cond_role_delay_ms"]
        for c in CONDS:
            s = t["online"]["lead1beat"][c]
            pd, ag = s["pc_diversity"], s["time_age_from_onset_ms"]
            out.append(f"| `{k}` | {t['targets']:,} / {m['notes']:,} ({pct(t['target_share_of_notes'])}) | {ex:,} | {rd['median']:.0f} · {rd['max']:.0f} | {c} | "
                       f"{s['mean_size']} | {pd['pc_set_size']['median']} | {pd['distinct_pc_sets']:,} | {pd['entropy_bits']} | {pct(pd['top5_share'])} | "
                       f"{ag['median']:.0f} · {ag['p90']:.0f} | {s['negative_d']} |")
    out += ["\n**연주자별 (주대상, 1박, 합산 비율)** — 곡 수가 적은 연주자는 몇 곡에 좌우된다\n",
            "| 연주자 | 곡 | g3 sounding | g3 recent | top50 sounding | top50 recent | grid300 sounding | grid300 recent |",
            "|---|---|---|---|---|---|---|---|"]
    for p, d in sorted(m["per_pianist_lead1beat"].items(), key=lambda x: (-x[1]["g3/sounding"]["songs"], x[0])):
        out.append(f"| {p} | {d['g3/sounding']['songs']} | " + " | ".join(pct(d[f'{k}/{c}']['nonempty_ratio']) for k in R for c in CONDS) + " |")
    for g, d in sorted(m["per_recording_type_lead1beat"].items()):
        out.append(f"| ({g}) | {d['g3/sounding']['songs']} | " + " | ".join(pct(d[f'{k}/{c}']['nonempty_ratio']) for k in R for c in CONDS) + " |")
    out += ["\n**A Fable 13곡 (탐색층, 자격 판정 없음)**\n",
            "| 규칙 | 타깃 | 온라인 K1/K2/K3 | replay·prefix·미래 교체 불일치 | sounding 1박 | recent 1박 | recent 곡별 최소 | 오프라인 1박 미래 onset 항목 |",
            "|---|---|---|---|---|---|---|---|"]
    for k in R:
        t, c = f["rules"][k], f["checks"][k]
        L = c["online_leaks"]
        s, rr = t["online"]["lead1beat"]["sounding"], t["online"]["lead1beat"]["recent"]
        out.append(f"| `{k}` | {t['targets']:,} | {L['K1']}/{L['K2']}/{L['K3']} | {c['replay_mismatch']}·{c['prefix_mismatch']}·{c['suffix_mismatch']} "
                   f"(표본 {c['samples']:,}) | {pct(s['nonempty_ratio'])} | {pct(rr['nonempty_ratio'])} | {pct(rr['per_song_nonempty_ratio']['min'])} | "
                   f"{pct(t['offline']['lead1beat']['entries_future_onset_share'])} |")
    return "\n".join(out)


def runtime_tables(r, ideal=None):
    m, f = r["strata"]["main"], r["strata"]["afable"]
    out = ["**검사 (주대상 189곡, 다섯 결정 모두, 전체 타깃)**\n",
           "| 규칙 | 타깃 | 온라인 K1/K2/K3 | replay 불일치 | prefix·미래 교체 표본 | prefix 불일치 | off 노출 | 미래 교체 불일치 | 구성 가능 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for k in R:
        v = m["verdicts"][k]
        L = v["online_leaks"]
        out.append(f"| `{k}` | {m['rules'][k]['targets']:,} | {L['K1']}/{L['K2']}/{L['K3']} | {v['replay_mismatch']} | {v['samples']:,} | "
                   f"{v['prefix_mismatch']} | {v['prefix_finite_off_exposed']} | {v['suffix_mismatch']} | {'예' if v['construction_ok'] else '아니오'} |")
    out += ["\n**비어 있지 않은 비율 (주대상)** — 판정은 위상 네 개 모두 1/10 이상(원래 개수)\n",
            "| 규칙 | 조건 | #1750 음별 1박 | 위상 0 | 위상 1 | 위상 2 | 위상 3 | 위상 최소 (원래 개수) | 최악 선행 1406.25 ms | 판정 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for k in R:
        for c in CONDS:
            on = m["rules"][k]["online"]
            ref = ideal["strata"]["main"]["rules"][k]["online"]["lead1beat"][c]["nonempty_ratio"] if ideal else None
            worst = min(PHASES, key=lambda p: on[p][c]["nonempty"] / on[p][c]["targets"] if on[p][c]["targets"] else 0)
            w = on[worst][c]
            out.append(f"| `{k}` | {c} | {pct(ref)} | " + " | ".join(pct(on[p][c]["nonempty_ratio"]) for p in PHASES)
                       + f" | {w['nonempty']:,}/{w['targets']:,} ({worst[-2:]}) | {pct(on['lead_worst'][c]['nonempty_ratio'])} | "
                       f"{m['verdicts'][k]['conditions'][c]['verdict']} |")
    out += ["\n**관측 간격과 조건 나이 (주대상)**\n",
            "| 규칙 | 결정 | primer 제외 | onset − d 중앙 · p90 · 최대 (ms) | sounding 조건 나이 d − on 중앙 · p90 | recent 평균 크기 | 곡별 sounding 최소 · 중앙 |",
            "|---|---|---|---|---|---|---|"]
    for k in R:
        t = m["rules"][k]
        for ln in PHASES + ("lead_worst",):
            ld, s = t["time_lead_ms"][ln], t["online"][ln]["sounding"]
            ag, ps = s["time_age_from_onset_ms"], s["per_song_nonempty_ratio"]
            out.append(f"| `{k}` | {ln} | {t['primer_excluded'][ln]:,} | {ld['median']:.0f} · {ld['p90']:.0f} · {ld['max']:.0f} | "
                       f"{ag['median']:.0f} · {ag['p90']:.0f} | {t['online'][ln]['recent']['mean_size']} | {pct(ps['min'])} · {pct(ps['median'])} |")
    out += ["\n**연주자별 (주대상, 위상 네 개 중 최소 비율)** — 곡 수가 적은 연주자는 몇 곡에 좌우된다\n",
            "| 연주자 | 곡 | g3 sounding | g3 recent | top50 sounding | top50 recent | grid300 sounding | grid300 recent |",
            "|---|---|---|---|---|---|---|---|"]
    pp = {p: m[f"per_pianist_{p}"] for p in PHASES}
    for name in sorted(pp["block_p0"], key=lambda n: (-pp["block_p0"][n]["g3/sounding"]["songs"], n)):
        cells = [pct(min(pp[p][name][f"{k}/{c}"]["nonempty_ratio"] for p in PHASES)) for k in R for c in CONDS]
        out.append(f"| {name} | {pp['block_p0'][name]['g3/sounding']['songs']} | " + " | ".join(cells) + " |")
    out += ["\n**A Fable 13곡 (탐색층, 자격 판정 없음)**\n",
            "| 규칙 | 온라인 K1/K2/K3 | replay·prefix·미래 교체 불일치 | sounding 위상 최소 | recent 위상 최소 | recent 곡별 최소(위상 0) |",
            "|---|---|---|---|---|---|"]
    for k in R:
        t, c = f["rules"][k], f["checks"][k]
        L = c["online_leaks"]
        mins = [min(t["online"][p][cc]["nonempty_ratio"] for p in PHASES) for cc in CONDS]
        out.append(f"| `{k}` | {L['K1']}/{L['K2']}/{L['K3']} | {c['replay_mismatch']}·{c['prefix_mismatch']}·{c['suffix_mismatch']} (표본 {c['samples']:,}) | "
                   f"{pct(mins[0])} | {pct(mins[1])} | {pct(t['online']['block_p0']['recent']['per_song_nonempty_ratio']['min'])} |")
    return "\n".join(out)


def main() -> None:
    r = json.load(open(sys.argv[1]))
    ideal = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
    print(runtime_tables(r, ideal) if r["params"].get("decision_set") == "runtime_block" else ideal_tables(r))


if __name__ == "__main__":
    main()
