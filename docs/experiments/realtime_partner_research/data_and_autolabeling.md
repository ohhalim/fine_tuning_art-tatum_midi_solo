# 재즈 피아노 chord/beat 라벨 데이터셋과 자동 라벨링 도구 (조사일 2026-10-11)

> 표기 규칙
> - **[직접 확인]**: 이번 세션에서 1차 출처(GitHub README/LICENSE/코드, 데이터 메타데이터 CSV, 기술 보고서 PDF 원문)를 직접 열어 확인함.
> - **[요약만]**: 웹 검색 결과 요약(검색 엔진이 생성한 요약/스니펫)만 확인. 원문 표/수치는 직접 대조하지 못함.
> - **[사용자 제공]**: 과제 브리핑에 있던 수치. 이번 세션에서 1차 출처로 재확인하지 못함.
> - 수치에 출처가 없으면 "미확인".
>
> 접근 환경 메모: 이 세션의 프록시에서는 arxiv.org / export.arxiv.org / huggingface.co / zenodo.org / transactions.ismir.net / archives.ismir.net / semanticscholar / github.io 가 모두 차단(curl CONNECT 403, WebFetch DNS 실패)되었고 **raw.githubusercontent.com 과 pypi.org 만 열림**. 따라서 GitHub에 올라온 README·LICENSE·코드·메타데이터 CSV·PDF(예: SheetSage2 기술 보고서가 YuE 저장소에 포함됨)는 [직접 확인], 논문 본문 수치 대부분은 [요약만]이다. 또 조사 중간에 세션 공용 WebSearch 한도(200회)가 소진되어 일부 항목(아래 Gaps)은 검색하지 못했다.
> 법적 해석은 법률 자문이 아니며, 라이선스 문구 자체만 인용한다.

---

## Q1. 데이터셋: 크기·악기·폴리포니·chord/beat 라벨·라이선스·접근 경로

### Takeaway
"재즈 **솔로 피아노** MIDI + 사람이 검수한 chord 라벨 + beat/downbeat 그리드"를 대규모로 공개한 데이터셋은 2026-10 현재 찾지 못했다. 가장 가까운 것은 (a) JTD(피아노 트리오, 자동 beat/downbeat + 자동 피아노 MIDI, chord 없음, MIT), (b) Filosax/FiloBass(사람 검수 beat·chord, 단 색소폰/베이스 중심이며 반주 오디오는 Aebersold 유료 구매), (c) JAAH(오디오 전용, 113곡, beat+chord, NC-SA), (d) QMUL의 "lead sheet 정렬 솔로 재즈 피아노" 대형 데이터셋(VAR4000은 그 부분집합; **미공개 상태로 보임**)이다. PiJAMA는 규모(2,777곡/223.6h)는 크지만 chord·beat 라벨이 없고 CC BY-NC 3.0이다.

### Cited Findings

**A. 재즈 피아노 MIDI(전사) 데이터**

- **PiJAMA** (Edwards, Dixon, Benetos, TISMIR 6(1):89–102, 2023, DOI 10.5334/tismir.162)
  - 메타데이터 CSV 직접 집계: 2,777 연주, 120명 피아니스트, 총 223.58시간(duration_sec 합), studio 1,994 / live 783, 모든 행에 `youtube_url` 컬럼 존재, split 컬럼 존재. Tigran Hamasyan 13곡, Brad Mehldau 72곡, Art Tatum 122곡. **[직접 확인]** — [pijama.csv](https://github.com/almostimplemented/PiJAMA/blob/main/pijama.csv)
  - LICENSE 파일 = "Creative Commons Attribution-NonCommercial 3.0 Unported". **[직접 확인]** — [PiJAMA LICENSE](https://github.com/almostimplemented/PiJAMA/blob/main/LICENSE)
  - MIDI는 Zenodo record 8354955에서 `midi_hawthorne.zip`로 배포(이름상 Kong et al. high-resolution transcription 계열 모델 사용을 시사하나 모델명은 README에 명시 안 됨). **[직접 확인]** — [PiJAMA README](https://github.com/almostimplemented/PiJAMA)
  - "over 200 hours", 244 albums, 120 pianists, Live/Studio 플레이리스트. **[요약만]** — [TISMIR 논문 페이지](https://transactions.ismir.net/articles/10.5334/tismir.162)
  - chord/beat 라벨: 없음(README·CSV 컬럼에 해당 필드 없음). **[직접 확인]** — [pijama.csv](https://github.com/almostimplemented/PiJAMA/blob/main/pijama.csv)
  - "고정 120 BPM 템포 맵으로 저장되어 실제 beat grid 없음": **[사용자 제공]**, 이번 세션 미확인.
  - 손(LH/RH) 분리: 메타데이터에 없음 **[직접 확인]**. Cheston 코드가 PiJAMA MIDI를 `instruments[0]` 단일 트랙으로 처리 **[직접 확인]** — [extractors.py](https://github.com/HuwCheston/deep-pianist-identification/blob/main/deep_pianist_identification/extractors.py) → 단일 트랙(손 미분리)일 가능성이 높음(추론).

- **Jazz Trio Database (JTD)** (Cheston, Schlichting, Cross, Harrison, TISMIR 2024, DOI 10.5334/tismir.186)
  - 1,294 트리오 연주(약 45h), 피아노 솔로 구간. 라벨: raw 믹스에 대한 **beat(4분음표) + downbeat**, 분리된 piano/bass/drums 각각에 onset·beat-matched onset, **피아노만 frame-level MIDI 전사**, 곡 단위 tempo·**박자(3 또는 4 quarter-note만)**·솔로 구간 타임스탬프·메타데이터. beat-onset 매칭 창은 −32분/+16분음표. **chord 라벨 없음**. **[직접 확인]** — [mirdata jtd.py](https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/jtd.py)
  - 라이선스: 주석·메타데이터 MIT, "YouTube 링크 오디오 사용은 이 라이선스가 다루지 않음"; 오디오(믹스/분리본)는 Zenodo에서 **연구 목적 접근 요청 후 승인**. **[직접 확인]** — [JTD README](https://github.com/HuwCheston/Jazz-Trio-Database), [jtd.py](https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/jtd.py)
  - 주석 파이프라인은 madmom으로 onset·beat·downbeat 추출, onset F-measure 평균 0.94(정답 대비), 44.5h. **[요약만]** — [TISMIR JTD](https://transactions.ismir.net/articles/10.5334/tismir.186)
  - beat F≈0.958, downbeat F≈0.408: **[사용자 제공]**, 이번 세션에서 원문 표 미확인.
  - 34명 피아니스트(Cheston 2025 인용). **[요약만]** — [arXiv 2504.05009](https://arxiv.org/html/2504.05009)

- **JAZZVAR** (arXiv 2307.09670, 2023; 이후 Springer 챕터로 출판): 솔로 피아노 재즈 스탠더드 연주에서 뽑은 502쌍의 'Original'(lead sheet)–'Variation'(연주 전사) MIDI 세그먼트. Variation은 오디오에서 전사 후 **deep chroma를 lead sheet chord symbol과 비교하는 Viterbi 디코딩으로 반자동 정렬**. 오디오 저작권 때문에 "직접 다운로드 불가" 상태. **[요약만]** — [arXiv 2307.09670](https://arxiv.org/pdf/2307.09670), [Zenodo 11264676](https://zenodo.org/records/11264676), [Springer](https://link.springer.com/chapter/10.1007/978-3-032-02042-0_8)

- **"lead sheet 정렬 솔로 재즈 피아노" 대형 데이터셋 (개발 중) / VAR4000**
  - Row, Shanin, Fazekas (QMUL), "Exploring Transformer-Based Music Overpainting for Jazz Piano Variations", ISMIR 2024 Late-Breaking Demo, arXiv 2412.04610 (2024-12-05): "솔로 피아노 재즈 연주를 lead sheet와 정렬한 대규모 구조화 데이터셋을 개발 중", head와 solo 구간 모두 정렬; 그 부분집합 **VAR4000 = 4,352 training pairs**. **[요약만]** — [arXiv 2412.04610](https://arxiv.org/abs/2412.04610)
  - 정렬 방법 원천: Shanin & Dixon, "Annotating Jazz Recordings Using Lead Sheet Alignment with Deep Chroma Features", **IEEE WASPAA 2023**(ISMIR 아님). **[요약만]** — [ResearchGate](https://www.researchgate.net/publication/374909877_Annotating_Jazz_Recordings_Using_Lead_Sheet_Alignment_with_Deep_Chroma_Features), [QMUL 목록](https://researchpublications.qmul.ac.uk/publications/staff/21674.html)
  - 검색 요약에는 "456 solo sections 고품질 전사", "Jazz Structure Dataset 340 recordings / 3000+ segments"라는 언급이 있었으나 출처 문맥이 불명확함 → 신뢰도 낮음. **[요약만]** — [arXiv 2412.04610 v1](https://arxiv.org/html/2412.04610v1)
  - 공개 여부: 2025–2026 공개 릴리스는 검색에서 찾지 못함(미확인).

- **Doug McKenzie 재즈 피아노 MIDI**: 한 학위논문(arXiv 1912.10815)이 "재즈 피아니스트가 직접 연주·녹음한 300곡 이상, 20시간 이상, 템포/다이내믹 변화가 잦고 화성이 복잡"한 컬렉션으로 기술. 라이선스/재배포 조건 미확인(웹사이트 이용약관 확인 필요). chord/beat 라벨 없음(미확인). **[요약만]** — [arXiv 1912.10815](https://arxiv.org/pdf/1912.10815)

**B. chord/beat 라벨이 있는 재즈 데이터(대부분 비피아노 또는 오디오 전용)**

- **JAAH** (Eremenko, Demirel, Bozkurt, Serra, ISMIR 2018): 113 재즈 녹음, beat 정렬 chord 시퀀스 + 구조 라벨, JSON/LAB 주석 + chroma feature, **오디오 미포함**. **[직접 확인: 저장소 구성]** — [MTG/JAAH](https://github.com/MTG/JAAH); 113곡·정렬 방식 **[요약만]** — [ISMIR 2018 PDF](https://archives.ismir.net/ismir2018/paper/000206.pdf), [Zenodo 1290737](https://zenodo.org/records/1290737)
  - 라이선스: ChoCo가 "JAAH 유래 데이터는 CC BY-NC-SA 4.0"으로 재배포한다고 명시. **[직접 확인]** — [ChoCo README](https://github.com/smashub/choco)
  - Beat This! 팀이 JAAH beat 주석을 교정하고 **downbeat 주석을 추가**해 공개. **[직접 확인]** — [beat_this_annotations](https://github.com/CPJKU/beat_this_annotations)
- **Weimar Jazz Database (WJazzD)**: ChoCo 파티션 표상 456 솔로(audio-aligned, Leadsheet 표기, SQL). **[직접 확인: 개수]** — [ChoCo README](https://github.com/smashub/choco). 모노포닉 솔로 전사 + 박절·chord·form·phrase 주석, 단일 SQLite 파일 배포. **[요약만]** — [Jazzomat DB 개요](https://jazzomat.hfm-weimar.de/dbformat/dboverview.html), [다운로드](https://jazzomat.hfm-weimar.de/download/download.html). 라이선스: 3자 카탈로그는 "Open Access"로만 표기, MeloSpySuite는 GPL. **ODbL 여부 미확인**. **[요약만]** — [MusoW 카탈로그](https://projects.dharc.unibo.it/musow/view-1645596749-4521604). 피아노 솔로 수(~6) **[사용자 제공]**, 미확인.
- **Filosax** (Foster & Dixon, ISMIR 2021): 곡당 ~6분, 7개 스템(Aebersold 반주 오디오 + **`Piano_Drums` 모노 믹스** + 색소폰 5명) 각 ~5시간. 주석: 마디 시작·마디 중간 chord 변화 지점 beat, 마디별 chord(JAMS), 노트 JSON에 chord_changes·scale degree 등. 다운로드: Zenodo에서 이용약관 동의 후 허가, **반주 오디오는 jazzbooks.com에서 구매**해 스크립트로 재구성. **[직접 확인]** — [Filosax README](https://github.com/dave-foster/filosax)
  - 라이선스 이중성: GitHub 저장소 LICENSE는 **CC0 1.0** **[직접 확인]**, 그러나 데이터 이용조건은 "서명한 개인과 그 연구그룹만 사용, 양도 불가, **비상업 연구 목적만**". **[직접 확인]** — [mirdata filosax.py LICENSE_INFO](https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/filosax.py)
- **FiloBass** (Riley & Dixon, ISMIR 2023, arXiv 2311.02023): Filosax 반주 기반 재즈 베이스 48곡 수동 검수 전사, 50,000+ 노트; 오디오 스템, MusicXML 악보, 연주 정렬 MIDI, **beat·downbeat·chord symbol·form 마커**. 라이선스 미확인(Zenodo 10265335). **[요약만]** — [arXiv 2311.02023](https://arxiv.org/pdf/2311.02023), [ISMIR 2023 포스터](https://ismir2023program.ismir.net/poster_236.html)
- **DTL1000** (Dig That Lick): 1920–2020 녹음, 1,060 오디오 트랙에서 1,750 즉흥연주; 메타데이터 + 수동 구조/스타일 주석 + **자동(CRNN) 모노포닉 주선율 전사**. beat/chord 라벨 언급 없음, 피아노 비율 불명, 라이선스 미확인. UK Data Service ReShare 854781. **[요약만]** — [UKDS ReShare](https://reshare.ukdataservice.ac.uk/854781/)

**C. 심볼릭 chord 차트(타이밍=마디/박 단위, 연주 아님)**

- **ChoCo** (de Berardinis et al.): 20,080 JAMS(오디오 2,283 / 심볼릭 17,803), chord 주석 20,530, beat 주석 286(오디오 파티션), 구조 554; 재즈 관련 파티션: JAAH 113(A), **The Real Book 2,486(S)**, WJazzD 456(A), **iReal Pro 2,000+(S)**, Jazz Corpus 76(S). 사람(전문가/검증된 크라우드) 주석만 포함. 라이선스 CC BY 4.0, 단 Chordify ASD·Mozart Piano Sonata·**JAAH 유래분은 CC BY-NC-SA 4.0**. **[직접 확인]** — [ChoCo README](https://github.com/smashub/choco)
- **Jazz Harmony Treebank** (Harasim et al., ISMIR 2020, DOI 10.5281/zenodo.4245406): iRealPro 코퍼스(Shanahan et al., Zenodo 3546040)에서 선택한 곡의 chord 시퀀스 + measures/beats + 계층 트리 분석 + 조성·박자. LICENSE = **CC BY-NC-SA 4.0**. 곡 수는 README에 명시 없음(미확인). **[직접 확인]** — [JazzHarmonyTreebank](https://github.com/DCMLab/JazzHarmonyTreebank)
- **iRb corpus**: OSU가 만든 "천 곡 이상의 재즈 스탠더드", 연구용 무료 공개라는 3자 설명만 확인; 1,186곡·Humdrum 형식·라이선스·iReal Pro 플레이리스트의 법적 지위는 미확인. **[요약만]** — [sharp11-irb (npm)](https://www.npmjs.com/package/sharp11-irb); 관련 선행: 기계가독 재즈 lead sheet DB(ISMIR 2013) **[요약만]** — [ISMIR 2013 PDF](https://archives.ismir.net/ismir2013/paper/000149.pdf)
- **Hooktheory (SheetSage 공개본)**: 50시간 분량 오디오 정렬 멜로디+화성 주석(JSON 20MB, 오디오 미포함), **CC BY-NC-SA 3.0**, artist 기준 8:1:1 split. 장르는 팝 위주. **[직접 확인]** — [SheetSage README](https://github.com/chrisdonahue/sheetsage)
- **Jazz lead sheet OMR (ISMIR 2025, arXiv 2509.05329)**: 손글씨 재즈 lead sheet 293장(163곡) + 정렬 심볼릭 정답, chord 인식 포함. 연주 데이터 아님. **[요약만]** — [ISMIR 2025 포스터](https://ismir2025program.ismir.net/poster_113.html), [arXiv 2509.05329](https://arxiv.org/html/2509.05329v1)
- 솔로 재즈 피아노 편곡에서 "lead sheet 화성"과 "편곡이 실제로 함의하는 화성"이 자주 다르다는 실무 교육자 설명(별도 두 종류 chord 라벨 제공). **[요약만]** — [Jeremy Siskind, Solo Jazz Standards](https://jeremysiskind.com/?p=14650)

**D. 비재즈 대형 피아노/멀티트랙 MIDI(사전학습·합성 라벨용)**

- **Aria-MIDI** (ICLR 2025): 1,186,253 파일, 약 100,629시간 솔로 피아노 전사, genre/composer/performer 메타데이터(텍스트 기반이라 정확도 가변), Full/Pruned(820,944)/Deduped(371,053)/Unique(32,522) 4종. **CC BY-NC-SA 4.0** + disclaimer 동의. 파이프라인 도구(Aria-AMT, Aria-CL)는 Apache-2.0. 재즈 곡 수 미확인. **[직접 확인]** — [aria-midi README](https://github.com/loubbrad/aria-midi)
- **GigaMIDI** (TISMIR 2025): 확장판 v2.0.0 2.1M+ 고유 MIDI, expressive loop 2.3M; **CC BY-NC 4.0**, HF에서 **접근 요청 필요**. **[직접 확인]** — [GigaMIDI README/LICENSE](https://github.com/Metacreation-Lab/GigaMIDI-Dataset)
- **GiantMIDI-Piano**: 10,855 파일/2,786 작곡가(클래식), disclaimer 동의 후 다운로드. 라이선스 문구는 이번에 확인 못함. **[직접 확인: 규모]** — [GiantMIDI-Piano](https://github.com/bytedance/GiantMIDI-Piano)
- **ATEPP** (ISMIR 2022): 11,674 연주(~1,000h), 49 피아니스트, 25 작곡가, 약 43% 악보 보유, **CC BY 4.0**(클래식). **[직접 확인]** — [ATEPP](https://github.com/BetsyTang/ATEPP)
- **POP909** (ISMIR 2020, arXiv 2008.07142): 팝 피아노 편곡, 곡별 beat(audio/midi), chord, key 파일, **MIT**. **[직접 확인]** — [POP909](https://github.com/music-x-lab/POP909-Dataset); 사람 교정판 POP909-CL(BACHI 논문). **[요약만]** — [arXiv 2510.06528](https://arxiv.org/html/2510.06528v2)
- **Lakh MIDI**: LMD-full 176,581 파일(MSD와 매칭, DTW 정렬 파이프라인). **[직접 확인]** — [craffel/midi-dataset](https://github.com/craffel/midi-dataset). 라이선스(CC BY 4.0로 알려져 있음)는 이번 세션 1차 출처 미확인.
- **Lakh 기반 chord 라벨(MidiCaps, arXiv 2406.02255)**: MIDI를 FluidSynth(FluidR3_GM)로 렌더 → **Chordino**로 chord 추출 → 가장 많이 반복되는 3/4/5-chord 시퀀스를 요약. 즉 "자동 라벨"이며 재즈 특화 아님. 라이선스 미확인. **[직접 확인: 코드]** — [MidiCaps pipeline.py](https://github.com/AMAAI-Lab/MidiCaps/blob/main/pipeline.py)
- **JazzNet** (ICASSP 2023, arXiv 2302.08632): 자동 생성 피아노 패턴 162,520개(chord·arpeggio·scale·progression, 88건반 전 조), ~95GB, 26K+ 시간 오디오, 패턴 생성기 제공, **MIT**. **[직접 확인]** — [jazznet](https://github.com/tosiron/jazznet)
- **PianoCoRe** (TISMIR, arXiv 2605.06627, 2026): "Combined and Refined Piano MIDI Dataset". PiJAMA 포함 여부·라이선스 미확인. **[요약만: 제목/서지]** — [arXiv 2605.06627](https://arxiv.org/pdf/2605.06627), [TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.333)

**E. PiJAMA 파생 연구(2024–2026)**

- **Edwards, Maezawa, Dixon, "Learning Jazz Pianist Style with Cross-Attention Conditioning", ISMIR 2026, arXiv 2610.02918**: Aria-medium + gated cross-attention + 12개 학습된 pianist embedding. PiJAMA-12 서브셋(Dick Hyman 187, **Art Tatum 118**, Abdullah Ibrahim 84, Dave McKenna 76, … Horace Parlan 28; **Mehldau·Hamasyan 미포함**). 분류기 95.8% chunk/98.8% track, 합성 전용 분류기 87.1/95.0, perplexity 6.82(cross-attn) vs 6.96 vs 11.41, 생성기 학습 약 18시간(24GB GPU 1장), 체크포인트 일부 "Apple-silicon 랩톱에서 실행" 명시. 코드 **Apache-2.0**; HF 체크포인트 라이선스 미확인. **chord 조건 없음**(아티스트 조건 연속 생성). **[직접 확인]** — [jazz-pianist-style README·data/pijama12.csv](https://github.com/almostimplemented/jazz-pianist-style); 서지 **[요약만]** — [arXiv 2610.02918](https://arxiv.org/abs/2610.02918)
- **Cheston, Bance, Harrison, "Deconstructing Jazz Piano Style Using Machine Learning", arXiv 2504.05009 (2025)**: PiJAMA + JTD로 피아니스트 식별, melody/harmony/rhythm/dynamics 4개 개념 분해. 코드 MIT. Cambridge 페이지는 Nature Machine Intelligence 게재 예정으로 표기 **[요약만]** — [Cambridge CMS](https://cms.mus.cam.ac.uk/publications/cheston-jazz-piano/). 화성 추출 정의는 Q3 참조 **[직접 확인]** — [deep-pianist-identification](https://github.com/HuwCheston/deep-pianist-identification)

### Inferences
- 사용자가 원하는 "chord + beat가 붙은 재즈 피아노 솔로 MIDI"는 공개 데이터로는 **조립해야** 한다. 현실적 재료: PiJAMA(MIDI·규모) + 자동 beat(Q2) + 자기라벨 chord(Q3) 또는 iReal/Real Book 차트(ChoCo·JHT) 정렬.
- Filosax 반주의 `Piano_Drums` 스템은 "사람 검수 beat + 마디별 chord가 붙은 재즈 피아노 컴핑 오디오"라는 드문 조합이다(오디오 전사 필요). 다만 Aebersold 유료 구매 + 서명 개인/연구그룹 한정·비상업 연구 조건이라 개인 취미 프로젝트가 "연구 목적"으로 승인받을 수 있을지는 불확실.
- JTD는 트리오 맥락이라 피아노 LH가 rootless 컴핑일 확률이 높아(베이스가 근음 담당) "LH PC-set = chord" 가정 검증용으로는 오히려 좋은 스트레스 테스트 세트가 된다(Q3).
- Tatum은 PiJAMA(122곡)와 Edwards PiJAMA-12(118곡)에 모두 있어 어댑터 데이터는 충분한 편이고, Mehldau(72곡)는 PiJAMA에만, Tigran(13곡)은 매우 적다.
- QMUL 그룹(Dixon/Fazekas/Shanin/Row/Edwards)이 lead-sheet 정렬 솔로 재즈 피아노 데이터셋을 만들고 있으므로, 공개 시점을 주기적으로 확인할 가치가 있다(현재는 미공개로 판단).

### Gaps
- PiJAMA "고정 120 BPM" 저장 방식, 전사 모델명·전사 정확도(논문 본문) — TISMIR/Zenodo 차단으로 미확인.
- JTD beat/downbeat F-measure 원문 표(사용자 제공 0.958/0.408) 미확인.
- WJazzD 라이선스(ODbL 여부)와 피아노 솔로 개수, DTL1000·FiloBass 라이선스, Doug McKenzie 이용약관, iRb/iReal Pro 플레이리스트의 법적 지위, Lakh MIDI 라이선스 문구 — 검색 한도 소진/차단으로 미확인.
- Charlie Parker Omnibook 정렬 데이터셋, Tigran Hamasyan 전사 데이터(공개 데이터셋 형태), "Jazz Piano Transcription" 정답 세트(사람 정렬)가 존재하는지 — 검색 한도 소진으로 조사 못함.
- Edwards/Cheston/Dixon의 "PiJAMA–lead sheet 정렬" 전용 연구는 찾지 못함(가장 근접: Shanin & Dixon WASPAA 2023, Row/Shanin/Fazekas 2024).
- Aria-MIDI·GigaMIDI 안의 재즈 피아노 비중(장르 메타데이터 집계) 미확인.

---

## Q2. 자동 라벨링 도구(beat, chord, lead-sheet 정렬, 손 분리)와 재즈에서의 측정 정확도

### Takeaway
재즈에서 **측정된** 수치가 있는 것은 주로 오디오 쪽이다: beat는 Beat This!가 JAAH(교차검증) beat F1 95.1 / downbeat 85.0 [요약만]으로 강하고, 오디오 chord는 SheetSage2-AR가 JAAH에서 maj/min 64.50, sevenths 47.89, **tetrads 43.00**(ChordFormer 40.04)으로 최고지만 9th/altered 텐션은 어휘에 없다 [직접 확인]. 심볼릭(MIDI) chord 인식(BACHI, AugmentedNet, ChordGNN)과 MIDI beat tracking(PM2S, Murgul&Heizmann)은 클래식/팝 위주 학습·평가이며 **재즈 수치 없음**. 손 분리·skyline도 재즈 수치를 찾지 못했다.

### Cited Findings

**A. Beat / downbeat (오디오)**

- **Beat This!** (Foscarin, Schlüter, Widmer, ISMIR 2024, arXiv 2407.21658): 코드와 **공개 가중치 모두 MIT**; 학습 데이터 일부는 저작권/제한적 CC라 사용자가 판단하라고 명시. 모델: final(약 78MB×3 seed), small(약 8.1MB×3), 8-fold 모델 등. `file2beats(audio)` → beats, downbeats. **[직접 확인]** — [beat_this README/LICENSE](https://github.com/CPJKU/beat_this)
  - 학습 16개 데이터셋에 **ASAP(클래식 피아노), Filosax, JAAH**(교정 beat + 추가 downbeat), GTZAN(테스트), HJDB, RWC 등 포함 → JAAH 평가는 in-domain. **[직접 확인]** — [beat_this_annotations](https://github.com/CPJKU/beat_this_annotations)
  - 8-fold CV 데이터셋별: **JAAH beat F1 95.1 / downbeat F1 85.0**, **Filosax 99.5 / 98.5**. continuity 지표는 DBN 계열보다 약함. **[요약만]** — [arXiv 2407.21658](https://arxiv.org/html/2407.21658v1); 후속 AlignBeat은 Beat This의 JAAH beat F 95.5, CMLt 88.5로 재보고(수치 차이 있음). **[요약만]** — [arXiv 2510.14391](https://arxiv.org/html/2510.14391)
  - SheetSage2 보고서의 GTZAN 재평가: Beat This beat F1 89.01, downbeat 78.28 (3 seed 평균, DBN 없음). **[직접 확인]** — [SheetSage2 기술 보고서 PDF](https://github.com/multimodal-art-projection/YuE/blob/main/docs/sheetsage2_technical_report.pdf)
- **madmom**: 소스 코드 BSD형, **모델/데이터 파일 CC BY-NC-SA 4.0**(pickled Processor 포함, 상업 사용 시 JKU 문의). **[직접 확인]** — [madmom LICENSE](https://github.com/CPJKU/madmom/blob/main/LICENSE). SheetSage 저자 평가: beat 검출은 "quite good", downbeat는 "brittle". **[직접 확인]** — [SheetSage README](https://github.com/chrisdonahue/sheetsage). GTZAN beat 86.07 / downbeat 64.65. **[직접 확인]** — [SheetSage2 보고서](https://github.com/multimodal-art-projection/YuE/blob/main/docs/sheetsage2_technical_report.pdf)
- **BeatNet** (Heydari et al., ISMIR 2021, arXiv 2108.03576): beat/downbeat/tempo/meter, **real-time·streaming·online·offline 모드**, CC BY 4.0, v1.2.0부터 학습 파이프라인 포함. 재즈 수치 미확인. **[직접 확인]** — [BeatNet](https://github.com/mjhydri/BeatNet)
- **All-In-One** (Kim & Nam, WASPAA 2023, arXiv 2307.16425): beat/downbeat/구조 분석, Demucs 분리 사용, Harmonix Set(팝) 8-fold 학습, MIT. 재즈 수치 미확인. **[직접 확인]** — [all-in-one](https://github.com/mir-aidj/all-in-one)
- 기타 2026 beat 연구: "Masked diffusion enables coherent beat tracking" (arXiv 2608.04624). 내용 미확인. **[요약만: 서지]** — [arXiv 2608.04624](https://arxiv.org/pdf/2608.04624)

**B. Beat / downbeat (심볼릭 MIDI 전용)**

- **PM2S** (Liu, Kong, Morfi, Benetos, ISMIR 2022): 노트 시퀀스 CRNN으로 beat/downbeat + key signature·time signature·**hand part** 예측. 코드 MIT. 학습 데이터 A-MAPS, Classical Piano MIDI, ASAP(모두 클래식). **[직접 확인: 라이선스·학습 데이터]** — [PM2S](https://github.com/cheriell/PM2S), [dev/README](https://github.com/cheriell/PM2S/tree/main/dev); 예측 항목·MV2H 우위 **[요약만]** — [ISMIR 2022 PDF](https://archives.ismir.net/ismir2022/paper/000047.pdf). 후속 연구는 개선판에서 time-signature 모듈이 제거되었고 4/4 고정이 결과를 개선했으며 beat-tracking 접근이 실제 녹음의 불규칙성에 약하다고 지적. **[요약만]** — [Beyer & Dai, arXiv 2410.00210](https://arxiv.org/html/2410.00210). 재즈 수치 없음.
- **Murgul & Heizmann, "Beat and Downbeat Tracking in Performance MIDI Using an End-to-End Transformer Architecture", SMC 2025, arXiv 2507.00466**: 인코더-디코더 Transformer, 동적 증강·토큰화, A-MAPS·ASAP·GuitarSet·Leduc에서 HMM/딥러닝 기준 대비 우수 주장. 구체 F1 수치 미확인. **[요약만]** — [arXiv 2507.00466](https://arxiv.org/abs/2507.00466), [KIT 목록](https://www.iiit.kit.edu/english/2148.php). 공식 저장소 README는 "Repo is still under construction!" 한 줄뿐(가중치 미공개). **[직접 확인]** — [Klangio/midi-beat-tracking](https://github.com/Klangio/midi-beat-tracking)
  - 같은 그룹의 beat 기반 리듬 양자화(arXiv 2508.19262): ASAP onset F1 97.3%, note value 정확도 83.3%. **[요약만]** — [arXiv 2508.19262](https://arxiv.org/pdf/2508.19262)

**C. Chord 인식 (오디오) — 재즈(JAAH) 수치**

- **SheetSage2** (Jiang, Yuan, Pan, Xue, Guo, Xia, LeCun; arXiv 2610.05336, 2026-10; 기술 보고서 2026-10-02): MERT-v2-FullSong 인코더, melody·chord·beat·downbeat·key·structure를 하나의 이벤트 시퀀스로 출력(ABC/MIDI). **[직접 확인]** — [기술 보고서 PDF](https://github.com/multimodal-art-projection/YuE/blob/main/docs/sheetsage2_technical_report.pdf), [YuE README](https://github.com/multimodal-art-projection/YuE)
  - **JAAH(113곡) chord maj/min**: SheetSage1 39.77, madmom 51.24, ChordFormer 59.45, SS2-Prober 62.94, **SS2-AR 64.50**. **[직접 확인]** (Table 2)
  - **JAAH 상세(Table 16, CF / Prober / AR)**: Root 60.43/66.62/67.55, Thirds 57.56/63.02/64.14, Triads 56.65/60.27/61.81, **Sevenths 46.49/45.15/47.89, Tetrads 40.04/40.22/43.00**, MIREX 63.57/69.83/70.68. **[직접 확인]** → 사용자가 언급한 "재즈 오디오 ACR ≈42%"는 tetrads 수준(40–43%)과 일치.
  - Chords1217(팝) maj/min 83.81(AR), GTZAN beat 86.27 / downbeat 80.45(AR). **[직접 확인]**
  - **어휘**: 361 상태 = N + 12 root × 30 quality-bass 조합; quality 15종(maj, min, dim, aug, maj7, min7, 7, hdim7, dim7, minmaj7, sus2, sus4, sus4(b7), maj6, min6) + 일부 전위. **9th 등 확장·변화 화음 표현 불가**. **[직접 확인]** (Table 12, §5)
  - **한계(원문)**: 곡 전체에 하나의 tempo anchor를 두고 상대 템포로 해석 → 큰 템포 변화에 부적합, **박자(meter) 인벤토리 제한**, steady-beat 팝 위주 학습, 표현적 템포의 클래식·free-rhythm 음악용 아님. **[직접 확인]** (§5)
  - 학습 데이터: 공개 심볼릭 데이터 MIDI 렌더링 + 라이선스된 Hooktheory + **비공개 lead-sheet 데이터** + HarmonixSet; AR은 CC0 음악과 (라이선스 하에 공급된) 합성 데이터 위주. **[직접 확인]** (Impact Statement)
  - 가중치·추론 코드: HF `m-a-p/SheetSage2`. **SheetSage2 가중치 라이선스는 미확인**(HF 차단). 같은 저장소의 YuE2 MODEL_LICENSE는 범위가 YuE2-3B/Vae 체크포인트로 명시되어 SheetSage2에 그대로 적용된다고 볼 수 없음(YuE2 가중치는 CC BY-NC 4.0 + 개인 창작자 수익화 허용 추가 허가). **[직접 확인]** — [YuE MODEL_LICENSE](https://github.com/multimodal-art-projection/YuE/blob/main/MODEL_LICENSE)
  - 3자 정보: 229MB, CPU 또는 4GB 미만 VRAM에서 동작 주장 **[직접 확인: 3자 README, 공식 아님]** — [RayCodes_SheetSage2](https://github.com/47thtechcorner/RayCodes_SheetSage2); 비공식 GGUF 변환 존재 **[요약만]** — [audio-cpp/SheetSage2-GGUF](https://huggingface.co/audio-cpp/SheetSage2-GGUF)
- **SheetSage (v1)** (Donahue, Thickstun, Liang, ISMIR 2022, arXiv 2212.01884): 코드 MIT, **전사 모델 CC BY-NC-SA 3.0**(Hooktheory 유래), madmom·Jukebox·Melisma 추가 라이선스. Linux+Docker(~4GB 컨테이너), ~24초 세그먼트 학습, Jukebox 특징은 12GB+ GPU 필요. JAAH maj/min 39.77(위 표). **[직접 확인]** — [SheetSage](https://github.com/chrisdonahue/sheetsage)
- **ChordFormer** (Akram et al., arXiv 2502.11840, 2025; SS2 보고서는 2026 출판으로 인용): Conformer + CQT, 대어휘 데이터에서 frame-wise +2%, class-wise +6% 주장. 라이선스·코드 공개 미확인. **[요약만]** — [arXiv 2502.11840](https://arxiv.org/html/2502.11840v1); JAAH 수치는 위 SS2 표 **[직접 확인]**.
- **JAAH 원 논문 베이스라인**: Chordino Jazz5 32.68 / MirexSevenths 24.57 / Tetrads 23.10, CREMA 40.26 / 37.54 / 34.30(표 열 정렬 모호, 검색 요약 기준). 저자들은 "Sevenths 어휘가 dim·6th 등을 제외해 재즈에 너무 빈약"하다고 지적. **[요약만]** — [ISMIR 2018 PDF](https://archives.ismir.net/ismir2018/paper/000206.pdf)
- **Chordino / NNLS Chroma**: GPL v2. **[직접 확인]** — [c4dm/nnls-chroma COPYING](https://github.com/c4dm/nnls-chroma)
- **BTC** (Park et al., ISMIR 2019): maj/min 또는 large-vocab 옵션, lab+MIDI 출력, MIT. 재즈 수치 미확인. **[직접 확인]** — [BTC-ISMIR19](https://github.com/jayg996/BTC-ISMIR19)
- **Durán & de la Cuadra, "Transcribing Lead Sheet-Like Chord Progressions of Jazz Recordings", Computer Music Journal 44(4):26–42, 2020**: 모든 코러스를 DTW로 정렬 후 CRNN으로 beat와 chord를 함께 추정, **재즈 녹음만으로 학습·평가**, 비재즈 대규모 학습 시스템보다 우수 주장; 코러스 시작/끝 시각을 입력으로 요구. 수치 미확인. **[요약만]** — [CMJ](https://direct.mit.edu/comj/article/44/4/26/108550/Transcribing-Lead-Sheet-Like-Chord-Progressions-of), [IEEE Xplore](https://ieeexplore.ieee.org/document/9655695)

**D. Chord / 기능화성 인식 (심볼릭)**

- **BACHI** (Yao, Chen, Dubnov, Berg-Kirkpatrick, UCSD; arXiv 2510.06528, v1 2025-10, v2 2026-02): beat-synchronous MIDI 토큰 입력, boundary 검출 + root/quality/bass 반복 디코딩, POP909-CL 공개, AugmentedNet·ChordGNN·Harmony Transformer v2와 비교, **팝·클래식만 평가(재즈 없음)**. ICASSP 2026 채택·MIT 라이선스는 [사용자 제공], 이번 세션 미확인. **[요약만]** — [arXiv 2510.06528](https://arxiv.org/html/2510.06528v2)
- **AugmentedNet** (Nápoles López, Gotham, Fujinaga, ISMIR 2021): Roman numeral 분석, MusicXML 입력, 합성 block-chord 텍스처화로 학습, MIT. 재즈 수치 없음. **[직접 확인]** — [AugmentedNet](https://github.com/napulen/AugmentedNet)
- **ChordGNN** (Karystinaios & Widmer, ISMIR 2023, arXiv 2307.03544): 노트 단위 GNN Roman numeral 분석, MIT. 재즈 수치 없음. **[직접 확인]** — [chordgnn](https://github.com/manoskary/chordgnn)
- 심볼릭 segmental CRF chord 인식(arXiv 1810.10002)도 클래식·팝 평가. **[요약만]** — [arXiv 1810.10002](https://arxiv.org/pdf/1810.10002)
- WJazzD를 이용한 심볼릭 chord 인식 벤치마크는 찾지 못함; Jazz Transformer(arXiv 2008.01307)는 WJazzD의 복잡한 chord 표기 처리 어려움을 언급(생성 연구). **[요약만]** — [arXiv 2008.01307](https://arxiv.org/pdf/2008.01307)

**E. Lead sheet ↔ 연주 정렬**

- Duan & Pardo (ISMIR 2011): 반즉흥 음악 오디오를 lead sheet에 chroma 기반 동적계획법으로 정렬(완전한 MIDI 악보 가정 불필요). **[요약만]** — [ISMIR 2011 PDF](https://archives.ismir.net/ismir2011/paper/000110.pdf)
- Shanin & Dixon (WASPAA 2023): deep chroma 기반 lead sheet 정렬로 재즈 녹음 주석화; JAZZVAR/VAR4000은 **deep chroma vs chord symbol + Viterbi** 정렬. 정렬 정확도 수치 미확인. **[요약만]** — [ResearchGate](https://www.researchgate.net/publication/374909877_Annotating_Jazz_Recordings_Using_Lead_Sheet_Alignment_with_Deep_Chroma_Features), [arXiv 2307.09670](https://arxiv.org/pdf/2307.09670)
- Durán & de la Cuadra(2020)의 코러스 간 DTW 정렬(위 C). **[요약만]**
- MIDI–오디오 DTW 정렬의 대규모 사례: Lakh 파이프라인(`align_text_matches.py`가 DTW로 정렬 + 진단 파일로 정합성 판정). **[직접 확인]** — [craffel/midi-dataset](https://github.com/craffel/midi-dataset)

**F. 손(LH/RH) 분리·멜로디 추출**

- PM2S가 노트별 hand part 예측(클래식 학습). 정확도 수치 미확인. **[요약만]** — [ISMIR 2022 PDF](https://archives.ismir.net/ismir2022/paper/000047.pdf)
- Hadjakos, Waloschek, Leemhuis (Mensch und Computer 2019 워크숍), "Detecting Hands from Piano MIDI Data": RNN으로 노트를 양손에 할당, 기존 휴리스틱(Kalman 계열)보다 우수, **실시간 동작·OSC 인터페이스**. 수치 미확인. **[요약만]** — [GI Digital Library](https://dl.gi.de/items/969b1f5c-6eb0-4761-a35c-ac1b067c4991)
- Skyline: 서양 클래식 피아노 멜로디 식별에서 최신 ML 대비로도 "매우 경쟁력"이라는 평가 **[요약만]** — [arXiv 2406.08454](https://arxiv.org/pdf/2406.08454); 팝에서는 반주가 멜로디 위에 오는 경우가 많아 실패, 다른 방법(Bayesian+DP)은 1,703마디에서 89% **[요약만]** — [SMC 2019 P1_10](https://www.smc2019.uma.es/articles/P1/P1_10_SMC2019_paper.pdf)
- Cheston의 MelodyExtractor: onset을 10ms로 양자화 후 같은 onset 그룹의 최고음만 남기는 skyline. **[직접 확인]** — [extractors.py](https://github.com/HuwCheston/deep-pianist-identification/blob/main/deep_pianist_identification/extractors.py)
- 2026 관련: "Statistical Models for Automatic Fingering-Annotated Piano Sheet Music Transcription" (arXiv 2609.28787) — 손/운지 관련 최신 연구, 내용 미확인. **[요약만: 서지]** — [arXiv 2609.28787](https://arxiv.org/html/2609.28787v1)

### Inferences
- **Beat grid**: PiJAMA처럼 드럼 없는 솔로 피아노(+Tatum 루바토)에 대한 재즈 측정치는 없다. Beat This는 ASAP(클래식 솔로 피아노)와 JAAH를 학습했으므로 현존 공개 도구 중 가장 유리하지만, downbeat는 beat보다 확실히 약할 것(JAAH 85.0 vs 95.1; JTD 0.408[사용자 제공])이고, 솔로 피아노 루바토에서는 더 떨어질 가능성이 크다(추론). SheetSage2는 "단일 tempo anchor·steady beat" 가정이라 Tatum 루바토·Tigran 변박엔 구조적으로 불리.
- **Chord 라벨**: 오디오 ACR 최고치도 JAAH tetrads 43% 수준이라 "7th 품질까지 맞는 chord 라벨"을 자동으로 얻으면 절반 이상이 틀린다. root(67.6%)·maj/min(64.5%) 수준의 거친 라벨 또는 연속형 chroma가 현실적이다.
- **심볼릭 경로**가 PiJAMA에는 더 자연스럽다(이미 MIDI가 있으므로). 그러나 BACHI·AugmentedNet·ChordGNN 모두 beat 위치/악보(MusicXML)를 전제하고 재즈 평가가 없으므로 "beat 추정 → 심볼릭 chord 인식" 2단 오류가 누적된다.
- **로컬 무료 우회안**: PiJAMA MIDI를 로컬에서 피아노 음원으로 렌더 → Beat This/SheetSage2 적용(MidiCaps가 Chordino에 FluidSynth 렌더를 쓰는 방식과 동일 패턴, SheetSage2 자체도 MIDI 렌더 합성 감독 사용). YouTube 다운로드 없이 beat를 얻을 수 있지만, 렌더 음색·전사 오류가 그대로 들어간다. 이 경우 시간축은 MIDI와 정확히 일치(렌더이므로).
- 실시간 파트너의 **사용자 입력** 쪽에서 LH/RH 분리가 필요하면 Hadjakos류 실시간 RNN 또는 단순 split-point가 후보; 학습 데이터 쪽 분리는 PM2S hand part가 유일한 공개 MIT 도구지만 재즈 검증 없음.

### Gaps
- Beat This의 데이터셋별 표(ASAP·HJDB·RWC Jazz 포함)와 최종 ISMIR판 수치 직접 대조 못함(arXiv 차단).
- Murgul&Heizmann F1 수치, PM2S hand-part 정확도, Hadjakos 손 분리 정확도, skyline의 재즈 피아노 정확도 — 1차 출처 차단으로 미확인.
- SheetSage2 가중치 라이선스(HF 모델 카드) 미확인; M1 Max에서의 실측 속도 미확인.
- 2024–2026 대어휘 chord 인식(ChordFormer·SheetSage2 외), 재즈 심볼릭 chord 인식 벤치마크 — 검색 한도 소진으로 추가 조사 못함.
- BACHI의 ICASSP 2026 채택·MIT 라이선스는 사용자 제공 정보로만 존재.

---

## Q3. 왼손 pitch-class set을 화성 라벨로 쓰는 것의 근거·위험, 대안, beat grid 확보 방법

### Takeaway
"LH PC-set을 chord 조건으로" 쓰는 자기라벨링은, 공개 연구 중 Cheston(2025)이 PiJAMA/JTD에서 쓴 화성 추출(동시음 ≥3개를 300ms 격자로 묶고 선택적으로 최고음 제거)이 가장 가까운 선례다(손 분리 없이 전체 음 사용). 실시간 파트너에서는 추론 시 조건이 "사용자 LH가 실제로 친 음"이므로 학습 조건도 같은 함수로 만드는 편이 train/test 일관성 측면에서 유리하다. 위험은 rootless voicing·stride 교대·손 교차 런·오스티나토로 인한 라벨 불안정과, 조건에 솔로 음이 섞여 모델이 조건을 복사하는 누수다. beat grid는 (1) YouTube 오디오에 Beat This 적용, (2) MIDI 렌더 후 Beat This, (3) MIDI 전용(PM2S) 중 택일이며 모두 재즈 솔로 피아노 검증 수치는 없다.

### Cited Findings
- Cheston HarmonyExtractor: "near-simultaneous" 음을 onset 기준 **0.3초(평균 JTD 템포의 4분음표)** 격자로 양자화해 묶고, **최소 3음 이상**(triad 이상)만 화성으로 인정, `remove_highest_pitch` 옵션으로 각 묶음의 최고음(멜로디)을 제거 가능; 손 분리는 하지 않음. **[직접 확인]** — [extractors.py](https://github.com/HuwCheston/deep-pianist-identification/blob/main/deep_pianist_identification/extractors.py)
- 같은 코드의 MelodyExtractor는 skyline(동일 onset 그룹 최고음)을 사용. **[직접 확인]** — [extractors.py](https://github.com/HuwCheston/deep-pianist-identification/blob/main/deep_pianist_identification/extractors.py)
- JTD 박자 라벨은 3 또는 4 quarter-note 박만 존재(변박 미지원). **[직접 확인]** — [mirdata jtd.py](https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/jtd.py)
- SheetSage2: 곡 단위 단일 tempo anchor, meter 인벤토리 제한, free-rhythm/표현적 템포 비대상, chord 어휘에 9th 없음. **[직접 확인]** — [SheetSage2 보고서](https://github.com/multimodal-art-projection/YuE/blob/main/docs/sheetsage2_technical_report.pdf)
- 재즈 오디오 chord 인식 최고치: JAAH tetrads 43.00 / sevenths 47.89 / root 67.55. **[직접 확인]** — [SheetSage2 보고서](https://github.com/multimodal-art-projection/YuE/blob/main/docs/sheetsage2_technical_report.pdf)
- 솔로 재즈 피아노 편곡에서 lead sheet chord와 편곡이 함의하는 chord가 자주 불일치하며, 함의 화성의 이름 붙이기에 정답이 하나가 아님. **[요약만]** — [Jeremy Siskind](https://jeremysiskind.com/?p=14650)
- 손 할당은 단순 휴리스틱보다 RNN이 정확(휴리스틱만으로는 오류가 남음을 시사). **[요약만]** — [Hadjakos et al. 2019](https://dl.gi.de/items/969b1f5c-6eb0-4761-a35c-ac1b067c4991)
- skyline은 반주가 멜로디 위에 오는 경우 실패(팝 사례). **[요약만]** — [SMC 2019](https://www.smc2019.uma.es/articles/P1/P1_10_SMC2019_paper.pdf)
- Beat This는 ASAP(클래식 솔로 피아노)와 JAAH·Filosax(재즈)를 학습 데이터에 포함, MIT 가중치. **[직접 확인]** — [beat_this_annotations](https://github.com/CPJKU/beat_this_annotations), [beat_this](https://github.com/CPJKU/beat_this)
- PiJAMA는 모든 곡에 YouTube URL과 연주 시작/끝 초를 제공(오디오–MIDI 대응 가능). **[직접 확인]** — [pijama.csv](https://github.com/almostimplemented/PiJAMA/blob/main/pijama.csv)
- MIDI 렌더 오디오에 오디오 chord 추출기를 거는 선례: MidiCaps(FluidSynth + Chordino). **[직접 확인]** — [MidiCaps pipeline.py](https://github.com/AMAAI-Lab/MidiCaps/blob/main/pipeline.py); SheetSage2도 MIDI 렌더 + MIDI 유래 라벨로 감독. **[직접 확인]** — [SheetSage2 보고서](https://github.com/multimodal-art-projection/YuE/blob/main/docs/sheetsage2_technical_report.pdf)
- PM2S(MIDI 전용 beat/downbeat/hand)는 클래식만 학습; 후속 연구는 실제 녹음 불규칙성에 약하고 4/4 고정이 낫다고 보고. **[직접 확인: 학습 데이터]** [PM2S dev](https://github.com/cheriell/PM2S/tree/main/dev); **[요약만: 비판]** [arXiv 2410.00210](https://arxiv.org/html/2410.00210)

### Inferences
- **rootless voicing(Evans/Kelly류 3-7-9-13, 7-3-13-9 등)**: 트리오(JTD)에서는 베이스가 근음을 담당하므로 LH PC-set에 근음이 빠져 서로 다른 chord가 같은 집합이 된다(예: 근음 없는 Dm9 ≈ Fmaj7 계열 집합). 솔로 피아노(PiJAMA)는 LH가 근음을 더 자주 치지만 피아니스트마다 다르다. → "root 라벨"로 해석하면 위험, "조건 = 실제로 울린 PC 집합"으로만 쓰면 문제가 줄어든다.
- **Tatum stride**: LH가 박 1·3 저음 단음 ↔ 박 2·4 중역 화음으로 교대 → 박 단위 PC-set이 박마다 1음/3음으로 흔들림. **반마디(또는 마디) 합집합 + 최저음(bass) 별도 슬롯**이 안정적. 양손 교차 런은 피치 분할선 기반 "LH"에 RH 런을 섞으므로, (i) 짧은 음 길이·빠른 연속음 필터, (ii) 동시성(Cheston식 ≥3음 묶음) 조건, (iii) PM2S hand part 등으로 거르는 처리가 필요.
- **Tigran 오스티나토/변박**: LH 리프가 페달·리듬 동기 역할을 해 PC-set이 경과음을 포함하고 화성 변화가 리프 주기와 어긋난다. 또한 7/8·5/4 등은 JTD(3/4·4/4만)·SheetSage2(meter 제한) 가정 밖이다. 13곡뿐이라 별도 수동 박 주석(탭)이 현실적.
- **누수 위험**: 조건을 "모든 음 chroma"로 만들면 솔로(RH) 음이 조건에 포함되어 모델이 조건을 베끼는 지름길을 학습할 수 있고, 추론 시(사용자 LH만 입력) 분포가 달라진다. → 조건은 LH/저역 또는 "최고음 제거 + ≥3음 동시성" 집합으로 제한하고, 생성 대상(RH/솔로 라인)과 겹치지 않게 분리.
- **권장 라벨 후보(정보량 순)**: ① bass-note(최저음) + PC-set(12-bit) per half-bar ② 12차원 duration-weighted chroma(LH/저역만) per beat ③ Cheston식 동시음 집합 ④ 차트 정렬(iReal/Real Book via ChoCo·JHT, 타이틀 매칭 31%[사용자 제공]) — ④는 "작곡 화성", ①–③은 "연주 화성"이며 실시간 파트너의 입력은 ①–③과 같은 성격이다.
- **beat grid 선택지**: (a) YouTube 오디오 + Beat This: 가장 정확할 가능성, 단 외부 다운로드(사용자 승인·YouTube 약관 문제) 필요, PiJAMA의 performance_start/end로 MIDI와 정렬. (b) PiJAMA MIDI 로컬 렌더 + Beat This: 외부 다운로드 불필요, MIDI와 시간축 자동 일치, 전사 오류·렌더 음색 영향. (c) PM2S MIDI 전용: 가장 간단하지만 클래식 학습·재즈 미검증. 어느 경우든 downbeat는 신뢰도가 낮으므로 "beat만 사용 + 마디 위상은 무시(또는 chord 변화 지점으로 사후 추정)" 설계가 안전.
- **실시간 조건과의 정합**: 실시간에서는 사용자의 LH가 MIDI로 바로 들어오므로 오디오 chord 인식이 필요 없다. 학습 라벨도 같은 "MIDI→PC-set" 함수를 쓰면 라벨 정확도(정답 chord)보다 **입력 분포 일치**가 확보된다는 점이 자기라벨링의 핵심 장점이다.

### Gaps
- LH PC-set을 chord 조건으로 쓴 재즈 생성 연구의 정량 평가(조건 일치율, 누수 정도)는 찾지 못함.
- rootless voicing이 chord 인식 정확도에 미치는 영향을 정량화한 논문은 이번 세션에서 찾지 못함(위 내용은 추론).
- 솔로 재즈 피아노(드럼 없음) 오디오에 대한 beat/downbeat 정확도 측정치 없음; JTD의 downbeat 0.408은 사용자 제공 수치.
- Tigran Hamasyan 곡의 박자/화성 주석 데이터는 찾지 못함.

---

## Q4. 항목별 확인 수준·라이선스와 개인 비상업 사용·공개 데모 영상에의 함의

### Takeaway
핵심 재료(PiJAMA CC BY-NC 3.0, Aria-MIDI CC BY-NC-SA 4.0, JHT CC BY-NC-SA 4.0, madmom 모델·SheetSage 모델·Hooktheory CC BY-NC-SA, GigaMIDI CC BY-NC 4.0, Filosax 서명형 비상업 연구)는 대부분 **비상업(NC)** 조건이라 개인 로컬 학습·비수익 데모까지는 문구상 허용 범위에 들지만, 광고 수익화 영상·모델/데이터 재배포는 위험하다. 반대로 **도구 쪽은 허용적**(Beat This MIT, PM2S MIT, BTC/AugmentedNet/ChordGNN/All-In-One MIT, BeatNet CC BY 4.0, JTD 주석 MIT)이라 라벨링 파이프라인 자체는 제약이 적다. SheetSage2 가중치 라이선스는 미확인이므로 사용 전 HF 모델 카드 확인이 필요하다.

### Cited Findings
| 항목 | 확인 수준 | 라이선스/접근 (출처) |
|---|---|---|
| PiJAMA MIDI | [직접 확인] | CC BY-NC 3.0, Zenodo 8354955 — [LICENSE](https://github.com/almostimplemented/PiJAMA/blob/main/LICENSE) |
| JTD 주석 | [직접 확인] | MIT; 오디오는 연구목적 요청 승인, YouTube 오디오는 라이선스 밖 — [README](https://github.com/HuwCheston/Jazz-Trio-Database) |
| JAAH | [직접 확인(ChoCo 경유)] | ChoCo 재배포분 CC BY-NC-SA 4.0, 오디오 미포함 — [ChoCo](https://github.com/smashub/choco) |
| ChoCo | [직접 확인] | CC BY 4.0(일부 NC-SA) — [ChoCo](https://github.com/smashub/choco) |
| Jazz Harmony Treebank | [직접 확인] | CC BY-NC-SA 4.0 — [JHT](https://github.com/DCMLab/JazzHarmonyTreebank) |
| Filosax | [직접 확인] | 저장소 CC0, 데이터는 서명 개인·연구그룹 한정 비상업 연구, 반주 유료 — [README](https://github.com/dave-foster/filosax), [mirdata](https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/filosax.py) |
| FiloBass | [요약만] | 미확인 (Zenodo 10265335) — [arXiv](https://arxiv.org/pdf/2311.02023) |
| WJazzD | [요약만] | 미확인("Open Access" 표기) — [MusoW](https://projects.dharc.unibo.it/musow/view-1645596749-4521604) |
| DTL1000 | [요약만] | 미확인 — [UKDS](https://reshare.ukdataservice.ac.uk/854781/) |
| JAZZVAR / VAR4000 | [요약만] | 오디오 저작권으로 직접 다운로드 불가 / 미공개 — [arXiv 2307.09670](https://arxiv.org/pdf/2307.09670), [arXiv 2412.04610](https://arxiv.org/abs/2412.04610) |
| Hooktheory(SheetSage) | [직접 확인] | CC BY-NC-SA 3.0 — [SheetSage](https://github.com/chrisdonahue/sheetsage) |
| Aria-MIDI | [직접 확인] | CC BY-NC-SA 4.0 + disclaimer — [aria-midi](https://github.com/loubbrad/aria-midi) |
| GigaMIDI | [직접 확인] | CC BY-NC 4.0, HF 접근 요청 — [GigaMIDI](https://github.com/Metacreation-Lab/GigaMIDI-Dataset) |
| ATEPP | [직접 확인] | CC BY 4.0 — [ATEPP](https://github.com/BetsyTang/ATEPP) |
| POP909 | [직접 확인] | MIT — [POP909](https://github.com/music-x-lab/POP909-Dataset) |
| JazzNet | [직접 확인] | MIT — [jazznet](https://github.com/tosiron/jazznet) |
| GiantMIDI / Lakh / MidiCaps | [직접 확인: 규모·방법] | 라이선스 문구 이번 세션 미확인 — [GiantMIDI](https://github.com/bytedance/GiantMIDI-Piano), [LMD](https://github.com/craffel/midi-dataset), [MidiCaps](https://github.com/AMAAI-Lab/MidiCaps) |
| Beat This! | [직접 확인] | 코드+가중치 MIT — [beat_this](https://github.com/CPJKU/beat_this) |
| madmom | [직접 확인] | 코드 BSD형, 모델 CC BY-NC-SA 4.0 — [LICENSE](https://github.com/CPJKU/madmom/blob/main/LICENSE) |
| BeatNet | [직접 확인] | CC BY 4.0 — [BeatNet](https://github.com/mjhydri/BeatNet) |
| All-In-One | [직접 확인] | MIT — [all-in-one](https://github.com/mir-aidj/all-in-one) |
| PM2S | [직접 확인] | MIT — [PM2S](https://github.com/cheriell/PM2S) |
| Murgul&Heizmann | [직접 확인: 저장소 상태] | 코드 미공개("under construction") — [Klangio](https://github.com/Klangio/midi-beat-tracking) |
| SheetSage v1 | [직접 확인] | 코드 MIT, 모델 CC BY-NC-SA 3.0 — [SheetSage](https://github.com/chrisdonahue/sheetsage) |
| SheetSage2 | [직접 확인: 보고서] / 라이선스 미확인 | HF m-a-p/SheetSage2; YuE2 가중치 라이선스(CC BY-NC 4.0+α)는 범위가 다름 — [보고서](https://github.com/multimodal-art-projection/YuE/blob/main/docs/sheetsage2_technical_report.pdf), [MODEL_LICENSE](https://github.com/multimodal-art-projection/YuE/blob/main/MODEL_LICENSE) |
| ChordFormer | [요약만] | 미확인 — [arXiv 2502.11840](https://arxiv.org/html/2502.11840v1) |
| Chordino/NNLS | [직접 확인] | GPL v2 — [nnls-chroma](https://github.com/c4dm/nnls-chroma) |
| BTC / AugmentedNet / ChordGNN | [직접 확인] | MIT — [BTC](https://github.com/jayg996/BTC-ISMIR19), [AugmentedNet](https://github.com/napulen/AugmentedNet), [ChordGNN](https://github.com/manoskary/chordgnn) |
| BACHI | [요약만] | MIT는 [사용자 제공], 미확인 — [arXiv 2510.06528](https://arxiv.org/html/2510.06528v2) |
| Edwards 2026 (jazz-pianist-style) | [직접 확인] | 코드 Apache-2.0, HF 체크포인트 라이선스 미확인 — [repo](https://github.com/almostimplemented/jazz-pianist-style) |
| Cheston DPI 코드 | [직접 확인] | MIT — [repo](https://github.com/HuwCheston/deep-pianist-identification) |

- Beat This README: "학습 파일 일부는 완전 저작권 또는 제한적 CC 라이선스이며 사용 사례 영향은 사용자가 판단". **[직접 확인]** — [beat_this](https://github.com/CPJKU/beat_this)
- SheetSage README: 모델이 Hooktheory 사용자 기여로 학습되어 CC BY-NC-SA 3.0이며 madmom·Jukebox·Melisma의 추가 조건이 상업 사용에 영향. **[직접 확인]** — [SheetSage](https://github.com/chrisdonahue/sheetsage)
- SheetSage2 Impact Statement: "전사된 lead sheet가 작품을 복제·배포할 권리를 주지 않으며 사용자는 권리·저작자 표시를 고려해야 한다". **[직접 확인]** — [보고서](https://github.com/multimodal-art-projection/YuE/blob/main/docs/sheetsage2_technical_report.pdf)
- madmom: pickled Processor(저장 모델)를 포함한 상업 제품 사용 시 JKU 연락 필요. **[직접 확인]** — [madmom LICENSE](https://github.com/CPJKU/madmom/blob/main/LICENSE)

### Inferences (법률 자문 아님)
- **개인 로컬 학습/실험**: NC 데이터(PiJAMA, Aria-MIDI, JHT 등)도 문구상 허용. ShareAlike(NC-SA)는 "각색물(라벨링된 데이터셋·파생 모델 가중치)을 공유할 때" 같은 라이선스를 요구하므로, 로컬 보관만 하면 실질 영향이 작다.
- **공개 데모 영상**: 비수익(광고·후원 연동 없음) 영상은 NC 조건과 충돌 가능성이 낮아 보이나, **YouTube 수익화는 '상업적'으로 해석될 여지**가 있어 피하는 편이 안전. 생성 MIDI가 PiJAMA 학습 모델의 출력이라는 점, 데모에서 재즈 스탠더드 멜로디/편곡을 연주하면 작곡 저작권 문제가 별도로 생길 수 있다는 점은 라이선스 문구만으로 판단 불가.
- **외부 다운로드 승인 우선순위(제안)**: 도구(Beat This MIT 가중치 → PM2S MIT → BTC/ChordGNN MIT)는 위험이 낮고, 데이터 중에는 JHT·ChoCo(심볼릭 차트, 소용량)가 PiJAMA 타이틀 정렬용으로 효율적. YouTube 오디오 대량 다운로드(PiJAMA 2,777곡)는 데이터 라이선스와 별개로 플랫폼 약관 문제가 있어 "MIDI 로컬 렌더 + Beat This" 대안을 먼저 검토할 가치가 있다.
- SheetSage2는 성능상 최선이지만 가중치 라이선스 미확인 + 비공개/라이선스 학습 데이터 사용이므로, 데모 영상에 결과를 쓰기 전에 HF 모델 카드 라이선스를 반드시 확인해야 한다.
- JTD 오디오는 "유효한 연구 프로젝트"에만 승인되므로 개인 프로젝트로 요청 시 거절 가능성을 고려(주석은 MIT로 자유).

### Gaps
- PiJAMA 등 NC 데이터로 학습한 **모델 가중치·생성물**의 라이선스 귀속 문제에 대한 데이터셋 저자 측 명시적 입장은 확인하지 못함.
- SheetSage2, ChordFormer, FiloBass, DTL1000, WJazzD, GiantMIDI, Lakh, MidiCaps, Edwards HF 체크포인트의 라이선스 원문 — 차단/검색 한도로 미확인.
- YouTube 오디오 다운로드의 약관상 허용 여부는 이번 조사 범위에서 1차 출처 확인 안 함.
