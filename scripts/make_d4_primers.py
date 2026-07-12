#!/usr/bin/env python3
"""D4용 conditioning 프라이머 변형 생성.

각 lead 샘플의 conditioning.mid(왼손 저음) + target.mid(오른손 리드)를 병합해
- twohand.mid : 두손 원본 텍스처 (H-D4a)
- chordal.mid : 동시타만 추린 화음 강조 프라이머 (H-D4a 세부)
를 만든다. 재학습 없이 조건 변수만 바꾸기 위한 도구.
"""
import argparse, glob, os
from collections import defaultdict
import pretty_midi


def load_notes(path):
    if not os.path.exists(path):
        return []
    pm = pretty_midi.PrettyMIDI(path)
    return [n for inst in pm.instruments for n in inst.notes]


def write_notes(notes, path, program=0):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    inst.notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    pm.instruments.append(inst)
    pm.write(path)


def make_chordal(notes, min_simul=2, window=0.05):
    """동시타(같은 onset ± window)가 min_simul개 이상인 노트만 남긴다."""
    groups = defaultdict(list)
    for n in notes:
        groups[round(n.start / window)].append(n)
    kept = []
    for g in groups.values():
        if len(g) >= min_simul:
            kept.extend(g)
    return kept if kept else notes  # 화음이 없으면 원본 유지


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles_dir", default="./data/roles/lead")
    ap.add_argument("--out_dir", default="./data/d4_primers")
    ap.add_argument("--num", type=int, default=24, help="샘플 수")
    ap.add_argument("--max_seconds", type=float, default=8.0,
                    help="프라이머로 쓸 앞부분 길이(초)")
    args = ap.parse_args()

    sample_dirs = sorted(glob.glob(os.path.join(args.roles_dir, "*")))[: args.num]
    os.makedirs(args.out_dir, exist_ok=True)
    n_two, n_chord = 0, 0
    for d in sample_dirs:
        sid = os.path.basename(d)
        left = load_notes(os.path.join(d, "conditioning.mid"))
        right = load_notes(os.path.join(d, "target.mid"))
        both = left + right
        if not both:
            continue
        # 앞부분만 자르기 (프라이머 길이 제한)
        both = [n for n in both if n.start < args.max_seconds]
        if not both:
            continue
        odir = os.path.join(args.out_dir, sid)
        os.makedirs(odir, exist_ok=True)
        write_notes(both, os.path.join(odir, "twohand.mid"))
        n_two += 1
        chordal = make_chordal(both)
        write_notes(chordal, os.path.join(odir, "chordal.mid"))
        n_chord += 1
    print(f"twohand={n_two} chordal={n_chord} out={args.out_dir}")


if __name__ == "__main__":
    main()
