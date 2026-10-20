## f28a96b8b6fb

**f28a96b8b6fb** over all 1251 programs: hazards-only verdict agreement 97.9% {'neither': 441, 'both': 784, 'cov-only': 16, 'vvl-only': 10}, all classes 95.4% {'neither': 230, 'both': 963, 'cov-only': 42, 'vvl-only': 16}; (mutant, finding) pairs all classes both/COVENANT-only/VVL-only = 2353 / 478 / 116; hazard pairs 1440 / 32 / 14

**f28a96b8b6fb+afu** over all 1251 programs: hazards-only verdict agreement 98.3% {'neither': 436, 'both': 794, 'cov-only': 21}, all classes 95.8% {'neither': 230, 'both': 969, 'cov-only': 42, 'vvl-only': 10}; (mutant, finding) pairs all classes both/COVENANT-only/VVL-only = 2367 / 511 / 102; hazard pairs 1454 / 65 / 0

| config | n | hazards: both / neither / COV only / VVL only | agree | all: both / neither / COV only / VVL only | agree |
|---|---|---|---|---|---|
| W1-F2 | 259 | 167 / 85 / 5 / 2 | 97.3% | 200 / 42 / 10 / 7 | 93.4% |
| W1-F3 | 259 | 153 / 97 / 8 / 1 | 96.5% | 189 / 59 / 7 / 4 | 95.8% |
| W2 | 239 | 151 / 84 / 1 / 3 | 98.3% | 183 / 43 / 10 / 3 | 94.6% |
| W3 | 245 | 168 / 72 / 2 / 3 | 98.0% | 203 / 34 / 7 / 1 | 96.7% |
| W2-MQ | 249 | 145 / 103 / 0 / 1 | 99.6% | 188 / 52 / 8 / 1 | 96.4% |
| ALL-SQ | 1002 | 639 / 338 / 16 / 9 | 97.5% | 775 / 178 / 34 / 15 | 95.1% |

| direction | class | reason | W1-F2 | W1-F3 | W2 | W3 | W2-MQ | total |
|---|---|---|---|---|---|---|---|---|
| cov-only | VAL-LIMIT | host-write race on mapped memory: syncval does not track host accesses (VVL docs/syncval_u | 31 | 34 | 48 | 22 | 70 | 205 |
| cov-only | TECH | same race reported a second time as a layout mismatch at the unordered present (COVENANT e | 18 | 26 | 24 | 27 | 19 | 114 |
| cov-only | TECH | same root cause (empty signal stage mask): COVENANT names the semaphore (signal-scope), sy | 16 | 14 | 15 | 11 | 20 | 76 |
| cov-only | DRIVER | acquire order diverges after a skipped present: the model driver's documented round-robin  | 21 | 30 | 0 | 0 | 0 | 51 |
| cov-only | MODEL | write-after-write through a barrier whose dst access mask is empty: COVENANT requires the  | 6 | 4 | 0 | 3 | 0 | 13 |
| cov-only | TECH | follow-on after a binary semaphore (acquire or render-finished) was signalled while a sign | 1 | 4 | 3 | 1 | 0 | 9 |
| cov-only | DRIVER | lavapipe hung or crashed before this frame (present:skip over-acquire, or re-recording a p | 1 | 3 | 0 | 0 | 0 | 4 |
| cov-only | VAL-LIMIT | EXCLUSIVE buffer read on the graphics family without the acquire half of the ownership tra | 0 | 0 | 0 | 0 | 3 | 3 |
| cov-only | TECH | follow-on of a non-increasing timeline signal (both report VUID-VkSubmitInfo2-semaphore-03 | 0 | 0 | 0 | 0 | 2 | 2 |
| cov-only | MODEL | write-after-write ordered only by an execution dependency (here a semaphore chain plus a b | 0 | 0 | 1 | 0 | 0 | 1 |
| vvl-only | TECH | VVL core reports the unwaited acquire as its own message; COVENANT reports the resulting r | 21 | 14 | 12 | 20 | 8 | 75 |
| vvl-only | DRIVER | surface minImageCount differs: lavapipe/X11 reports 3 (1 image may be held), the model dri | 12 | 11 | 0 | 0 | 0 | 23 |
| vvl-only | MODEL | layout transition not ordered after the acquire of an image that has not been presented ye | 4 | 1 | 3 | 5 | 1 | 14 |
| vvl-only | MODEL | first use of a never-presented swapchain image without waiting for the acquire: COVENANT a | 2 | 0 | 1 | 0 | 1 | 4 |

**stored cov.fix -> f28a96b8b6fb:** verdict changed (all classes) 0, (hazards) 0; normalised key sets changed 0; diagnostics changed in code/where 106, text only 435; warnings changed 104; delta {'gained timing.barrier-scope': 61, 'gained timing.signal-stage-scope': 180, 'lost timing.barrier-scope': 384}

**stored cov.snap -> f28a96b8b6fb:** verdict changed (all classes) 138, (hazards) 129; normalised key sets changed 464; diagnostics changed in code/where 494, text only 305; warnings changed 104; delta {'gained timing.acquire-semaphore-in-use': 512, 'gained timing.barrier-scope': 282, 'gained timing.command-buffer-in-use': 690, 'gained timing.cross-queue-race': 130, 'gained timing.fence-in-use': 72, 'gained timing.missing-barrier': 8, 'gained timing.semaphore-resignal': 306, 'gained timing.signal-stage-scope': 563, 'gained timing.timeline-value': 13, 'lost timing.barrier-scope': 384}

**archived -> f28a96b8b6fb:** verdict changed (all classes) 0, (hazards) 0; normalised key sets changed 0; diagnostics changed in code/where 0, text only 0; warnings changed 0; delta {}

**guarantee eligibility** (tools/check_trace.py stats.eligibility): {'W1-F2': {'eligible': 259}, 'W1-F3': {'eligible': 259}, 'W2': {'eligible': 239}, 'W3': {'eligible': 245}, 'W2-MQ': {'eligible': 249}}; not eligible 0; unknown 0; tool vs timing-meta mismatches 0; check_trace errors 0

triage rows changed vs stored fix (f28a96b8b6fb): 0; unexplained: 0

triage rows changed vs stored fix (f28a96b8b6fb+afu): 25; unexplained: 0
