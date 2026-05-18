# 실시간 AI 재즈 즉흥 시스템 선행연구 종합 (2026-02-20 기준)

이 글은 내가 조사한 선행연구자/논문/팀을 한 번에 정리한 메모다.
목표는 명확하다: **실시간 MIDI 즉흥 + 특정 피아니스트 스타일 학습**을 공연 가능한 수준으로 구현하는 것.

---

## 1) 내가 풀고 싶은 문제

- 입력: FL Studio + MCP + MIDI 키보드 코드/제스처
- 모델: 심볼릭(토큰) 기반 Transformer
- 출력: 실시간 솔로 MIDI
- 최종 목적: 하우스/테크노 드롭 구간에서 스타일 솔로를 라이브로 안정 출력

핵심 제약:
- 지연(latency)과 지터(jitter)를 라이브 수준으로 낮춰야 함
- 스타일 충실도(예: Brad Mehldau-like)를 유지해야 함
- 공연 중 실패 대비 fallback이 있어야 함

---

## 2) 선행연구자 맵 (내 프로젝트 관점)

## A. 심볼릭 음악 생성 핵심 라인

1. **Cheng-Zhi Anna Huang**
- Music Transformer 핵심 저자
- 최근 실시간 협업/적응 반주 연구 라인까지 연결됨

2. **Douglas Eck**
- Magenta 창립 및 음악 생성 연구 리더십

3. **Yi-Hsuan Yang**
- 스타일 조건부 심볼릭 생성/전이 연구 라인에서 중요한 축

## B. 실시간 인간-AI 즉흥 상호작용 라인

1. **François Pachet**
- Continuator로 실시간 스타일 반응 시스템의 고전적 기반 제시

2. **Lancelot Blanchard / Perry Naseck / Stephen Brade / Kimaya Lecamwasam / Joseph Paradiso**
- JAM_BOT 시스템 구현/공연 검증 축
- 논문에서 “모델 성능”보다 “공연 가능한 시스템 설계”를 전면에 둠

## C. 스타일/데이터 기반 연구 연결점

1. **Simon Dixon / Emmanouil Benetos**
- PiJAMA 같은 자동전사 재즈 피아노 심볼릭 데이터셋 맥락에서 참고 가치 큼

2. **David Cope**
- AI 스타일 작곡의 초기 역사 축

## D. 산업 SOTA 추적 라인 (오디오 생성)

1. **DeepMind Lyria/Realtime**
- 실시간 생성, 제어성, 시스템화 측면에서 벤치마크 가치가 큼

2. **Suno / Udio / Stability AI**
- 최종 음원 품질 관점의 최신 제품 라인
- 단, Suno는 공개 논문/저자 정보가 제한적이라 회사 공지 중심 추적 필요

---

## 3) 핵심 논문/프로젝트 요약

| 구분 | 작업 | 핵심 기여 | 내 프로젝트에 주는 시사점 |
|---|---|---|---|
| Symbolic | Music Transformer (2018) | 장기 구조를 다루는 심볼릭 Transformer 기반 | 내 기본 모델 축의 출발점 |
| Symbolic | Anticipatory Music Transformer (2024) | 조건/미래 문맥 활용 강화 | chord-conditioned 제어성 강화에 직접 참고 |
| Realtime | ReaLChords (ICML 2024) | RL 기반 실시간 적응 반주 | 라이브 강건성(낯선 입력 대응) 기준점 |
| Realtime | ReaLJam (2025) | 인간-AI 실시간 잼 상호작용 | 내 최종 사용 시나리오와 매우 유사 |
| Realtime System | JAM_BOT (ISMIR 2025) | 역할 분리, 멀티스레드, 저지연 생성 스케줄러 | “역할별 모델 + 오케스트레이션 레이어” 접근 정당화 |
| HCI/Creative | Developing Symbiotic Virtuosity (2024) | AI-연주자 공진화 설계 | 기술지표 + 무대 경험의 동시 최적화 필요 |
| Historical RT | Continuator (2003) | 스타일 기반 실시간 반응의 고전 | 저복잡도 fallback 전략 설계 참고 |

---

## 4) JAM_BOT에서 내가 배운 설계 포인트

1. **단일 모델 만능주의를 버림**
- 역할(lead/accompaniment/call-response) 분리

2. **프롬프트 전략을 분리**
- 정해진 바 교대
- 제스처 기반 재프롬프트
- 온디맨드 트리거

3. **데이터를 즉흥 “녹음”이 아니라 “학습 과제”로 구조화**
- conditioning 신호와 target 신호를 분리해 수집

4. **실시간 엔진은 멀티스레드 구조가 핵심**
- 입력, 처리, 생성, 스케줄링 분리

5. **가속은 필수**
- quantization, KV cache, 컨텍스트 축소, 선생성(1~2 bar ahead)

---

## 5) 회사/팀별로 추적할 포인트

## DeepMind (Lyria/Realtime)
- 실시간성 + 제어성 + 연구 공개라는 점에서 학습 가치가 가장 높음

## Suno
- 결과 음원 품질은 매우 강함
- 공개된 연구자/논문 정보가 상대적으로 제한적
- 회사/투자사/공식 블로그 공지 기반으로 사람과 방향 추적

## Udio / Stability AI
- 오디오 생성 제품화 및 성능 확장 흐름 추적에 유효
- 내 프로젝트는 note-level 실시간 제어가 핵심이므로 심볼릭 라인을 주축으로 유지

---

## 6) 내 프로젝트에 바로 적용할 실천 원칙

1. Stage A는 **성공확률 우선**: lead 1역할 + 최소 conditioning + 안정 동작
2. RunPod 학습, 로컬 추론 분리
3. KPI 고정: TTFN, dead-air, chord-change response delay
4. role-conditioned 데이터셋을 기본 형식으로 강제
5. 스타일 충실도 문제는 모델 크기보다 데이터 정합성부터 점검
6. 실시간 실패 대비 fallback phrase 즉시 전환 규칙 내장
7. 논문 구현보다 공연 안정성을 1순위로 둠

---

## 7) 결론

내 결론은 단순하다.

- “최고 품질 단일 모델”을 찾는 문제보다,
- **실시간 연주 가능한 시스템 아키텍처**를 먼저 완성하는 게 맞다.

즉, 지금 단계의 정답은:
- `pretrained symbolic model + role-conditioned fine-tuning + low-latency runtime`

그리고 그 다음에:
- 역할 추가, 조건 토큰 확장, ONNX/int8/KV cache로 점진 최적화.

---

## References

- Music Transformer: https://research.google/pubs/music-transformer-generating-music-with-long-term-structure/
- AMT (Anticipatory Music Transformer): https://openreview.net/forum?id=EBNJ33Fcrl
- JAM_BOT DOI: https://doi.org/10.5281/zenodo.17811480
- JAM_BOT Zenodo: https://zenodo.org/records/17706584
- JAM_BOT ISMIR page: https://ismir2025program.ismir.net/poster_321.html
- Developing Symbiotic Virtuosity: https://mit-genai.pubpub.org/pub/iz684jjr
- Continuator page (Pachet): https://www.francoispachet.fr/publications/
- DeepMind Lyria: https://deepmind.google/models/lyria/
- DeepMind Lyria Realtime: https://deepmind.google/technologies/lyria/realtime/
- Magenta Realtime repo: https://github.com/magenta/magenta-realtime
- Suno series C: https://about.suno.com/blog/series-c-announcement
- Suno acquires WavTool: https://suno.com/blog/suno-acquires-wavtool
- Udio launch PR: https://www.prnewswire.com/news-releases/former-google-deepmind-researchers-assemble-luminaries-across-music-and-tech-to-launch-udio-a-new-ai-powered-app-that-allows-anyone-to-create-extraordinary-music-in-an-instant-302113166.html
- Stability Audio 2.5: https://stability.ai/news/stability-ai-introduces-stable-audio-25-the-first-audio-model-built-for-enterprise-sound-production-at-scale
- MusicGen: https://arxiv.org/abs/2306.05284
- MusicLM: https://arxiv.org/abs/2301.11325
- AudioLM: https://arxiv.org/abs/2209.03143
