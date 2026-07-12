# Cold email draft — Prof. Anna Huang (LMDM / real-time symbolic music)

> 목적: 실시간 AI 음악 협연 연구자로서 접점 만들기. LMDM(Live Music Diffusion Models)
> 저자 그룹. 짧게, 구체적으로, 내 진단 실험 한 건을 미끼로.
> 톤: 정중하되 팬레터 아님 — 동료 연구자로서 하나의 구체적 발견을 공유.
> 보내기 전 [대괄호] 부분을 본인 상황에 맞게 채우세요.

---

**Subject:** A conditioning-primer texture-lock finding in symbolic jazz-piano generation

Dear Professor Huang,

I'm [이름], a jazz pianist working on real-time AI musical co-improvisation —
a symbolic MIDI model that generates jazz piano in interaction with a human player.
Your work on Live Music Diffusion (the KV-cache routing and ARC-Forcing for
low-latency block generation) has been a central reference for my latency budget.

I wanted to share one finding, because it turned out to be a clean instance of a
problem your conditioning work touches. Fine-tuning a from-scratch Music Transformer
to a single pianist's style collapsed generation diversity, which I first assumed was
a base-capacity limit. A controlled experiment ruled that out: holding the model and
hyperparameters fixed and only widening the training data (16 → 2,777 pieces) recovered
diversity entirely — the small model had already reached 93% of its *training data's*
diversity ceiling. The bottleneck was the data regime, not expressiveness.

Chasing the residual monophony further, I isolated the real cause to the **conditioning
primer itself**. A lead-only (right-hand) primer was locking the model into a monophonic
texture: swapping only the conditioning to unconditional lifted mean voicing size from
1.31 to 1.62, essentially back to the two-hand source ceiling (1.66) — the model could
always play chords, but the primer told it not to. The conditioning implicitly fixes not
just *what* to play but *what texture* to play in, which seems directly relevant to how
live conditioning shapes generation in your setting.

I'm still early and doing this alongside a separate career track, but I'd value any
pointer — a paper, a direction, or a five-minute reaction to whether this texture-lock
framing is something your group has run into. Either way, thank you for making the
LMDM work open; it's been genuinely useful.

Best regards,
[이름]
[한 줄 소속/링크 — GitHub repo나 짧은 소개 페이지]

---

## 보내기 전 체크리스트

- [ ] Anna Huang 현재 소속·정확한 이메일 확인 (개인 조사)
- [ ] [이름]/[소속] 채우기, GitHub repo 링크 추가 (진단 실험을 볼 수 있게)
- [ ] Subject를 본인 실험 한 건으로 좁게 유지 (일반적 "collaboration" 제목 피하기)
- [ ] 수치는 이 초안이 실제 실험값 그대로 (93% 상한 도달률, voicing 1.31→1.62→1.66)
- [ ] 첨부보다 링크. 답장 오면 그때 세부 공유
