#!/bin/zsh
# On-device latency table (macOS): every configuration in its OWN process over the timing sample, --gap s between frames.
# usage: SAMPLE=sample.json FRAMES=<frames root> LORA=<v1 pixel LoRA .pt | none> OCR_LORA=<OCR LoRA .pt | none> MODE=busy \
#          scripts/run_timing_table.sh [config ...]
#        configs: full full+pe S S+T S+T+L S+G S+G+T+L ocr-rapid ocr-vision (default: all); OCR rows use OCR_LORA and never --compile
#        MODE=busy starts scripts/load/load.sh (Speedometer + 1080p video) and waits 60 s; MODE=idle runs nothing else.
#        CPU rows: DEVICE=cpu DTYPE=float32 COMPILE=1 WSTORE=bf16-llm-fp32 (idle, 32 GB) or WSTORE=bf16 (in the wild, 16 GB laptop);
#        OCR rows on the CPU then use WSTORE=bf16 (stock forward)
#        NCHUNKS=2 ... = smoke test (40 frames)
# Before an idle run: reboot, quit apps, AC power, Low Power Mode off. Output: $OUT/<MODE>_<name>.json (+ _emb.npy, _fids.npy)
set -e
ROOT=${0:A:h:h}; cd "$ROOT"
PY=${PY:-python}; MODE=${MODE:-busy}; OUT=${OUT:-results_timing}; mkdir -p "$OUT"
: ${SAMPLE:?set SAMPLE}; : ${FRAMES:?set FRAMES}; : ${LORA:?set LORA (path or none)}
CFGS=("$@"); [ ${#CFGS} -eq 0 ] && CFGS=(full full+pe S S+T S+T+L S+G S+G+T+L ocr-rapid ocr-vision)
WSTORE=${WSTORE:-native}
COMMON=(--sample "$SAMPLE" --frames-root "$FRAMES" --out "$OUT" --tag "$MODE" --gap ${GAP:-0.3}
        --device ${DEVICE:-mps} --dtype ${DTYPE:-float16} --threads ${THREADS:-8}
        --mps-cap-gb ${MPS_CAP_GB:-9.5} --chunk-step ${CHUNK_STEP:-1} --n-chunks ${NCHUNKS:-0} --min-avail-gb ${MIN_AVAIL_GB:-0.3})
# changed: macOS Command Line Tools without libc++ headers next to clang -> torch.compile (inductor) fails with "'algorithm' file not found"
SDKCXX=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
[ "${COMPILE:-0}" = 1 ] && [ -d $SDKCXX ] && export CPLUS_INCLUDE_PATH=$SDKCXX
if [ "$MODE" = busy ]; then
  caffeinate -d -i -w $$ &                       # keep the display on: a sleeping display throttles the browser / video load
  scripts/load/load.sh start; sleep 60           # let the foreground load reach steady state
fi
for c in $CFGS; do
  echo "=== $MODE $c $(date '+%F %T')"
  if [[ $c == ocr-* ]]; then                     # OCR rows: OCR LoRA, stock forward (no --compile, no LLM-fp32 storage)
    : ${OCR_LORA:?set OCR_LORA (OCR LoRA path or none)}
    ARGS=(--route $c --lora "$OCR_LORA" --weight-storage ${WSTORE/bf16-llm-fp32/bf16})
  else
    ARGS=(--preset $c --lora "$LORA" --weight-storage $WSTORE); [ "${COMPILE:-0}" = 1 ] && ARGS+=(--compile)
  fi
  caffeinate -i $PY -m screenret.timing $ARGS $COMMON 2>&1 | grep -v -E "Loading weights|objc\[|onnxruntime|unauthenticated" || true
  sleep 30                                       # cool-down between configurations
done
[ "$MODE" = busy ] && scripts/load/load.sh stop
echo "ALL DONE $MODE $(date '+%F %T')"
