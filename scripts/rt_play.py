#!/usr/bin/env python3
"""Play with the realtime left-hand path from a real keyboard (issue #1826, docs/RT_PLAY.md).

Left-hand MIDI input (a keyboard or FL Studio) -> rt_lh_producer.py (Aria, q8 by default) -> rt_sender.py -> MIDI output
(an existing IAC bus or a virtual port that FL Studio / Serum listens to). An optional beat click with a count-in goes to
its own virtual port. Ctrl-C stops the session: the producer is interrupted, the sender sees the end of its input and
cancels unsent note-ons, turns off every sounding note and closes the port. Each session is logged under
rt_play_sessions/<timestamp>/. The block contract is fixed at 128 BPM (aria_mlx_stream_loop BEAT / BLOCK).

  rt_play.py --list
  rt_play.py [--lh PORT] [--out PORT] [--minutes M | --blocks N] [--precision q8|fp32] [--click] [--count-in BEATS]
  rt_play.py --dry-run [--blocks N]     (virtual ports, a short fake left hand, output captured; no keyboard needed)

PORT is a 1-based index from --list or a unique part of the name. Without --lh / --out the ports are asked for.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from block_contract import BEAT, BLOCK  # noqa: E402  (128 BPM, the producer's contract)

GIT_BOX = "/Users/ohhalim/git_box"
DEFAULTS = {"t3_py": f"{GIT_BOX}/t3_aria/.venv/bin/python", "aria_repo": f"{GIT_BOX}/t3_aria/aria",
            "ckpt": f"{GIT_BOX}/t3_aria/ckpt/model-gen.safetensors", "data": f"{GIT_BOX}/mehldau_aria/data.json",
            "sessions": f"{GIT_BOX}/rt_play_sessions"}
BPM = 60_000 / BEAT
CLICK_PORT = "rt_play click"
CLICK_NOTES = (76, 77)                 # GM hi / low wood block on channel 10: bar start, other beats


def display_name(raw):
    """CoreMIDI names with non-ASCII characters can come back from rtmidi decoded as Mac Roman ("IAC ÎìúÎùºÏù¥Î≤Ñ");
    undo that for display. The raw name is still what opens the port."""
    try:
        fixed = raw.encode("mac_roman").decode("utf-8")
        return fixed if fixed != raw else raw
    except (UnicodeEncodeError, UnicodeDecodeError):
        return raw


def pick_port(names, choice):
    """A port from a 1-based index or a unique, case-insensitive part of its raw or display name."""
    if not names:
        raise ValueError("no MIDI ports")
    c = str(choice).strip()
    if c.isdigit():
        i = int(c)
        if 1 <= i <= len(names):
            return names[i - 1]
        raise ValueError(f"index {i} not in 1..{len(names)}")
    hits = [n for n in names if c.lower() in n.lower() or c.lower() in display_name(n).lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError(f"no port matches {c!r}")
    exact = [n for n in hits if display_name(n).lower() == c.lower() or n.lower() == c.lower()]
    if len(exact) == 1:
        return exact[0]
    raise ValueError(f"{c!r} matches {len(hits)} ports: " + ", ".join(display_name(n) for n in hits))


def blocks_for(minutes):
    """Blocks for a session of `minutes` (plus the cold block 0)."""
    return int(math.ceil(minutes * 60_000 / BLOCK)) + 1


def lead_for(count_in_beats):
    """Wall seconds from the end of the cold block to the music origin: room for the count-in, never under the
    producer's default 1.0 s."""
    return max(1.0, round(count_in_beats * BEAT / 1000 + 0.25, 3))


def non_negative_int(text):
    """argparse type: an integer >= 0 (a negative note cap would push every onset to the next window, #1841 review)."""
    v = int(text)
    if v < 0:
        import argparse
        raise argparse.ArgumentTypeError(f"0 이상이어야 함: {v}")
    return v


def non_negative_float(text):
    """argparse type: a number >= 0 (density tilt strength, #1844)."""
    import math
    v = float(text)
    if not math.isfinite(v) or v < 0:
        import argparse
        raise argparse.ArgumentTypeError(f"0 이상의 유한한 수여야 함: {v}")
    return v


def producer_cmd(cfg, files):
    return [cfg["t3_py"], os.path.join(HERE, "rt_lh_producer.py"), cfg["aria_repo"], cfg["ckpt"], cfg["data"],
            str(cfg["blocks"]), cfg["lh_port"], files["origin"], files["progress"], files["producer"],
            "--precision", cfg["precision"], "--lead-s", str(cfg["lead_s"])] + \
        (["--note-cap", str(cfg["note_cap"])] if cfg.get("note_cap") else []) + \
        (["--density-tilt", str(cfg["density_tilt"]), "--density-n0", str(cfg["density_n0"])]
         if cfg.get("density_tilt") else [])                                        # off unless asked (#1844)


def sender_cmd(cfg, files):
    cmd = [cfg["py"], os.path.join(HERE, "rt_sender.py"), cfg["out_port"], files["ready"], files["sender"]]
    return cmd + (["--existing-port"] if cfg["out_existing"] else [])


def click_times(wall_origin_ns, count_in_beats, total_beats):
    """(wall ns, accent) for every beat from the count-in up to the end; bar starts are counted from the origin."""
    beat_ns = BEAT * 1e6
    return [(int(wall_origin_ns + b * beat_ns), b % 4 == 0) for b in range(-count_in_beats, total_beats)]


def fake_lh(blocks, offset_ms=BLOCK):
    """A short left hand for the dry run: one shell voicing per block (2 beats), Dm7 G7 Cmaj7 A7, held 1.5 beats."""
    voicings = ((50, 53, 57, 60), (43, 53, 55, 59), (48, 52, 55, 59), (45, 49, 55, 57))
    notes = []
    for k in range(max(0, blocks - 2)):
        on = k * BLOCK
        for p in voicings[k % 4]:
            notes.append([p, round(on), round(on + 1.5 * BEAT), 70])
    return notes


def summarize(progress_rows, producer, sender, capture=None):
    """One record of the session from what the processes wrote (any of them may be missing after a crash)."""
    warm = [r for r in progress_rows if r.get("block", 0) >= 1 and r.get("ready_margin_ms") is not None]
    margins = sorted(r["ready_margin_ms"] for r in warm)
    out = {"blocks_timed": len(warm), "blocks_committed": sum(1 for r in warm if not r.get("problem")),
           "deadline_misses": sum(1 for m in margins if m < 0),
           "margin_min_ms": margins[0] if margins else None,
           "margin_median_ms": margins[len(margins) // 2] if margins else None,
           "abort": (producer or {}).get("abort") or next(({"reason": r["problem"], "block": r["block"]} for r in progress_rows
                                                            if r.get("problem")), None),
           "crops": (producer or {}).get("crops"), "lh_notes_received": ((producer or {}).get("lh_counts") or {}).get("on")}
    if sender:
        out.update(sender_end=sender.get("end"), notes_on_sent=sum(1 for r in sender.get("sent", []) if r["kind"] == "on"),
                   rejected_late_on=len(sender.get("rejected_late_on", [])), pending_on_at_stop=sender.get("pending_on_at_stop"),
                   sounding_at_stop=sender.get("sounding_at_stop"), stop_offs_failed=sender.get("stop_offs_failed"),
                   port_closed=sender.get("port_closed"), clean_stop=sender.get("clean_stop"))
    if capture is not None and sender:
        out["captured_messages"] = len(capture.get("messages", []))
        out["sent_messages"] = len(sender.get("sent", []))
    return out


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass                   # a line cut by an interrupt
    return rows


def read_json(path):
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return None


def clicker(port, times, stop):
    """Send the click on time until stop is set; short notes on channel 10."""
    import mido
    for t, accent in times:
        while time.monotonic_ns() < t:
            if stop.is_set():
                return
            time.sleep(min(0.002, max(0.0, (t - time.monotonic_ns()) / 1e9)))
        note = CLICK_NOTES[0] if accent else CLICK_NOTES[1]
        port.send(mido.Message("note_on", channel=9, note=note, velocity=110 if accent else 75))
        time.sleep(0.03)
        port.send(mido.Message("note_off", channel=9, note=note, velocity=0))


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def _child_signals():
    """Children start with the default SIGINT even when rt_play was launched with it ignored (a background job)."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def stop_process(p, sig, wait_s):
    """Send sig (None: none, the signal was already sent) and wait; terminate, then kill, if it does not exit."""
    if p is None or p.poll() is not None:
        return
    try:
        if sig is not None:
            p.send_signal(sig)
        p.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def shutdown(procs, interrupted, stop, sender_drain_s):
    """Stop order (Astra review of #1833). A user stop (Ctrl-C, SIGTERM): the click and the sender stop first and at the
    same time as the producer is signalled, so the sender does not keep sending scheduled note-ons while a slow producer
    is waited for; each child gets SIGINT once (a second one could cut the sender's note-off cleanup). A normal end: the
    producer has finished, the sender plays out what is scheduled, then stops."""
    sender, producer = procs.get("sender"), procs.get("producer")
    if interrupted:
        stop.set()
        for p in (sender, producer):
            if p is not None and p.poll() is None:
                try:
                    p.send_signal(signal.SIGINT)
                except Exception:
                    pass
        stop_process(sender, None, 15)
        stop_process(producer, None, 5)
    else:
        stop_process(producer, signal.SIGINT, 5)
        if sender is not None and sender.poll() is None:
            try:
                sender.wait(timeout=sender_drain_s)
            except subprocess.TimeoutExpired:
                pass
        stop.set()
        stop_process(sender, signal.SIGINT, 15)
    stop_process(procs.get("feeder"), signal.SIGTERM, 5)


def progress_line(r):
    if r.get("problem"):
        return f"  블록 {r.get('block')}: 중단 ({r['problem']})"
    if r.get("block") == 0:
        return f"  블록 0(시작 전 생성): {r.get('cold_s')} s"
    return f"  블록 {r['block']:>3}  여유 {r['ready_margin_ms']:+7.1f} ms  음 {r.get('notes')}  왼손 {r.get('lh_notes')}"


def run(cfg):
    import mido
    d = cfg["session_dir"]
    os.makedirs(d, exist_ok=True)
    f = {k: os.path.join(d, v) for k, v in {"origin": "origin.json", "progress": "producer_progress.jsonl", "producer": "producer.json",
                                             "perr": "producer.stderr", "sender": "sender.json", "serr": "sender.stderr",
                                             "ready": "capture.ready", "stop": "capture.stop", "capture": "capture.json",
                                             "feeder": "feeder.json", "fready": "feeder.ready", "lh_notes": "fake_lh.json"}.items()}
    with open(os.path.join(d, "config.json"), "w") as fo:
        json.dump({k: v for k, v in cfg.items()}, fo, ensure_ascii=False, indent=1)
    procs, click_port, stop = {}, None, threading.Event()
    interrupted = False
    try:
        if cfg["dry_run"]:
            json.dump(fake_lh(cfg["blocks"]), open(f["lh_notes"], "w"))
            procs["capture"] = subprocess.Popen([cfg["py"], os.path.join(HERE, "rt_capture.py"), cfg["out_port"], f["ready"], f["stop"],
                                                 f["capture"], str(int(cfg["blocks"] * BLOCK / 1000) + 120)])
            procs["feeder"] = subprocess.Popen([cfg["py"], os.path.join(HERE, "rt_lh_feeder.py"), cfg["lh_port"], f["lh_notes"],
                                                f["origin"], f["fready"], f["feeder"]])
            t = time.monotonic()
            while not os.path.exists(f["fready"]) and time.monotonic() - t < 30:
                time.sleep(0.02)
        else:
            open(f["ready"], "w").write(str(time.monotonic_ns()))     # no capture in a live session
        procs["sender"] = subprocess.Popen(sender_cmd(cfg, f), stdin=subprocess.PIPE, stderr=open(f["serr"], "w"),
                                           start_new_session=True, preexec_fn=_child_signals)
        procs["producer"] = subprocess.Popen(producer_cmd(cfg, f), stdout=procs["sender"].stdin, stderr=open(f["perr"], "w"),
                                             start_new_session=True, preexec_fn=_child_signals)
        procs["sender"].stdin.close()
        print("모델 로딩과 시작 전 블록 생성 중...", flush=True)
        while not os.path.exists(f["origin"]):
            if procs["producer"].poll() is not None:
                raise RuntimeError("producer가 시작 전에 끝남. producer.stderr 확인")
            time.sleep(0.05)
        time.sleep(0.05)
        origin = read_json(f["origin"])
        W = origin["wall_origin_ns"]
        if cfg["click"]:
            click_port = mido.open_output(CLICK_PORT, virtual=True)
            times = click_times(W, cfg["count_in"], int((cfg["blocks"] - 1) * BLOCK / BEAT))
            threading.Thread(target=clicker, args=(click_port, [x for x in times if x[0] > time.monotonic_ns()], stop),
                             daemon=True).start()
        wait = (W - time.monotonic_ns()) / 1e9
        print(f"{max(0.0, wait):.1f}초 뒤 시작 ({BPM:.0f} BPM, 블록 {cfg['blocks'] - 1}개 ≈ {(cfg['blocks'] - 1) * BLOCK / 60000:.1f}분)."
              f" 모델은 시작 시각부터 솔로, 왼손 반영은 다음 블록부터. Ctrl-C로 정지", flush=True)
        shown = 0
        while procs["producer"].poll() is None:
            rows = read_jsonl(f["progress"])
            for r in rows[shown:]:
                print(progress_line(r), flush=True)
            shown = len(rows)
            time.sleep(0.2)
        for r in read_jsonl(f["progress"])[shown:]:
            print(progress_line(r), flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\n정지 중: 소리 나는 음 끄는 중...", flush=True)
    finally:
        shutdown(procs, interrupted, stop, 2 * BLOCK / 1000 + 10)
        if "capture" in procs:
            open(f["stop"], "w").close()
            try:
                procs["capture"].wait(timeout=30)
            except subprocess.TimeoutExpired:
                procs["capture"].kill()
        if click_port is not None:
            try:
                for n in CLICK_NOTES:
                    click_port.send(mido.Message("note_off", channel=9, note=n, velocity=0))
                click_port.close()
            except Exception:
                pass
    summary = summarize(read_jsonl(f["progress"]), read_json(f["producer"]), read_json(f["sender"]),
                        read_json(f["capture"]) if cfg["dry_run"] else None)
    summary.update(interrupted=interrupted, session_dir=d)
    with open(os.path.join(d, "summary.json"), "w") as fo:
        json.dump(summary, fo, ensure_ascii=False, indent=1)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="실제 키보드 왼손으로 실시간 경로 연주 (docs/RT_PLAY.md)")
    ap.add_argument("--list", action="store_true", help="MIDI 포트 목록")
    ap.add_argument("--lh", help="왼손 입력 포트(번호 또는 이름 일부)")
    ap.add_argument("--out", help="출력 포트(기존 포트 번호·이름 일부, 또는 --out-virtual 이름)")
    ap.add_argument("--out-virtual", help="기존 포트 대신 이 이름의 가상 출력 포트를 만듦")
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--blocks", type=int, help="블록 수(시작 전 블록 포함). --minutes보다 우선")
    ap.add_argument("--precision", default="q8", choices=("q8", "fp32", "bf16"))
    ap.add_argument("--note-cap", type=non_negative_int, default=0, help="2박 블록 창당 음 수 상한(#1819). 0이면 끔(기본)")
    ap.add_argument("--density-tilt", type=non_negative_float, default=0.0, help="밀도 기울기 강도(#1844). 0이면 끔(기본)")
    ap.add_argument("--density-n0", type=non_negative_int, default=8, help="기울기가 시작되는 창당 음 수(#1844)")
    ap.add_argument("--click", action="store_true", help=f"박 클릭을 가상 포트 '{CLICK_PORT}'로(채널 10)")
    ap.add_argument("--count-in", type=int, default=4, help="클릭 카운트인 박 수")
    ap.add_argument("--dry-run", action="store_true", help="가상 포트와 가짜 왼손으로 연결 확인")
    for k in ("t3_py", "aria_repo", "ckpt", "data", "sessions"):
        ap.add_argument("--" + k.replace("_", "-"), default=DEFAULTS[k])
    a = ap.parse_args(argv)
    signal.signal(signal.SIGINT, signal.default_int_handler)       # Ctrl-C works even if started with SIGINT ignored
    signal.signal(signal.SIGTERM, _interrupt)                       # a kill stops cleanly too
    import mido
    ins, outs = mido.get_input_names(), mido.get_output_names()
    if a.list:
        print("입력 포트(왼손):")
        for i, n in enumerate(ins, 1):
            print(f"  {i}. {display_name(n)}")
        print("출력 포트(모델 솔로 → DAW):")
        for i, n in enumerate(outs, 1):
            print(f"  {i}. {display_name(n)}")
        return 0
    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = {"py": sys.executable, "t3_py": a.t3_py, "aria_repo": a.aria_repo, "ckpt": a.ckpt, "data": a.data,
           "precision": a.precision, "note_cap": a.note_cap, "density_tilt": a.density_tilt, "density_n0": a.density_n0, "click": a.click, "count_in": a.count_in, "dry_run": a.dry_run,
           "blocks": a.blocks or (9 if a.dry_run else blocks_for(a.minutes)), "bpm": BPM}
    cfg["lead_s"] = lead_for(a.count_in if a.click else 0)
    if a.dry_run:
        cfg.update(lh_port=f"rt_play_dry_lh_{os.getpid()}", out_port=f"rt_play_dry_out_{os.getpid()}", out_existing=False)
    else:
        lh = a.lh or input("왼손 입력 포트 번호 또는 이름: " + ", ".join(f"{i}. {display_name(n)}" for i, n in enumerate(ins, 1)) + "\n> ")
        cfg["lh_port"] = pick_port(ins, lh)
        if a.out_virtual:
            cfg.update(out_port=a.out_virtual, out_existing=False)
        else:
            o = a.out or input("출력 포트 번호 또는 이름: " + ", ".join(f"{i}. {display_name(n)}" for i, n in enumerate(outs, 1)) + "\n> ")
            cfg.update(out_port=pick_port(outs, o), out_existing=True)
    cfg["session_dir"] = os.path.join(a.sessions, ("dry_" if a.dry_run else "") + tag)
    print(f"왼손 {display_name(cfg['lh_port'])} → Aria {cfg['precision']} → {display_name(cfg['out_port'])}. 기록 {cfg['session_dir']}")
    s = run(cfg)
    print("요약:", json.dumps({k: v for k, v in s.items() if k != "session_dir"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
