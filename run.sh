#!/usr/bin/env bash
# Build and run all smoke tests, then write results/report.txt.
# Usage: ./run.sh -DGPU_BACKEND=CUDA|HIP|CUSTOM [other CMake options]
# Environment: BUILD_DIR (default build), OUT_DIR (default results), TEST_TIMEOUT (seconds per test, default 300),
#              STACK_BYTES (stack per thread for the recursion tests, default 16384)
set -u
cd "$(dirname "$0")"
BUILD=${BUILD_DIR:-build}
OUT=${OUT_DIR:-results}
TIMEOUT=${TEST_TIMEOUT:-300}
STACK=${STACK_BYTES:-16384}
rm -rf "$OUT"
mkdir -p "$OUT"
R="$OUT/results.txt"
: > "$R"
log() { echo "$*" | tee -a "$R"; }

# ---- 08: build integration (every target is built separately, so one failure does not hide the others)
if cmake -S . -B "$BUILD" "$@" > "$OUT/configure.log" 2>&1; then
  log "RESULT 08.build.configure PASS cmake configure"
else
  log "RESULT 08.build.configure FAIL cmake configure, see configure.log"
fi
for t in t01_virtual t02_link t03_recursion t04_image t05_fp64 t06_atomics t07_runtime; do
  if cmake --build "$BUILD" --target "$t" > "$OUT/build-$t.log" 2>&1; then
    log "RESULT 08.build.$t PASS built"
  else
    log "RESULT 08.build.$t FAIL see build-$t.log"
    if grep -q '__cxa_pure_virtual' "$OUT/build-$t.log"; then
      log "RESULT 01.virtual.pure_virtual FAIL no device __cxa_pure_virtual for abstract classes (retry with -DGPU_PURE_VIRTUAL_SHIM=ON)"
    fi
  fi
done

# run <log name> <required|info> <executable> [args]
run() {
  local name=$1 level=$2 exe="$BUILD/bin/$3" id
  id=$(echo "$1" | sed 's/_/./; s/_/./') # 03_recursion_d64 -> 03.recursion.d64
  shift 3
  if [ ! -x "$exe" ]; then
    log "RESULT $id.run FAIL not built"
    return
  fi
  timeout "$TIMEOUT" "$exe" "$@" > "$OUT/$name.log" 2>&1
  local rc=$? bad=FAIL
  [ "$level" = info ] && bad=INFO
  grep '^RESULT' "$OUT/$name.log" | tee -a "$R"
  if [ $rc -eq 124 ]; then
    log "RESULT $id.exit $bad timeout after ${TIMEOUT}s"
  elif [ $rc -gt 128 ]; then
    log "RESULT $id.exit $bad killed by signal $((rc - 128))"
  elif ! grep -q '^RESULT' "$OUT/$name.log"; then
    log "RESULT $id.exit $bad no result printed (exit code $rc)"
  fi
}

run 01_virtual required 01_virtual
run 02_link required 02_link
for d in 8 64 256; do
  run "03_recursion_d$d" info 03_recursion "$d" 0
  run "03_recursion_d${d}_stack" required 03_recursion "$d" "$STACK"
done
run 04_image required 04_image
run 05_fp64 required 05_fp64
run 06_atomics required 06_atomics
run 07_runtime required 07_runtime
if grep -q 'device printf ok' "$OUT/07_runtime.log" 2> /dev/null; then
  log "RESULT 07.runtime.printf_output PASS device printf output reached stdout"
else
  log "RESULT 07.runtime.printf_output WARN device printf output not seen"
fi
# A device trap in its own process: the host should get an error, not hang
if [ -x "$BUILD/bin/07_runtime" ]; then
  timeout 120 "$BUILD/bin/07_runtime" trap > "$OUT/07_trap.log" 2>&1
  rc=$?
  if grep -q '^RESULT' "$OUT/07_trap.log"; then
    grep '^RESULT' "$OUT/07_trap.log" | tee -a "$R"
  elif [ $rc -eq 124 ]; then
    log "RESULT 07.runtime.trap WARN the host hangs after a device trap"
  else
    log "RESULT 07.runtime.trap WARN the runtime aborts the process on a device trap (exit code $rc)"
  fi
fi

# ---- report: one line per row of the requirements table
status() {
  local lines f w p
  lines=$(grep -E "^RESULT ($1)" "$R")
  f=$(echo "$lines" | awk '$3 == "FAIL"' | grep -c .)
  w=$(echo "$lines" | awk '$3 == "WARN"' | grep -c .)
  p=$(echo "$lines" | awk '$3 == "PASS"' | grep -c .)
  if [ "$f" -gt 0 ]; then
    echo "FAIL ($f failed)"
  elif [ "$p" -eq 0 ]; then
    echo "NOT RUN"
  elif [ "$w" -gt 0 ]; then
    echo "PASS ($w optional item(s) missing)"
  else
    echo "PASS"
  fi
}
cache() { grep -E "^$1:" "$BUILD/CMakeCache.txt" 2> /dev/null | head -1 | cut -d= -f2-; }
{
  echo "GPU portability smoke tests - report"
  echo "Date:     $(date -u '+%Y-%m-%d %H:%M UTC')"
  echo "Host:     $(uname -srm)"
  echo "Backend:  $(cache GPU_BACKEND)$([ "$(cache GPU_PURE_VIRTUAL_SHIM)" = ON ] && echo ' (with GPU_PURE_VIRTUAL_SHIM)')"
  for lang in CUDA HIP CXX; do
    comp=$(cache "CMAKE_${lang}_COMPILER")
    [ -n "$comp" ] && echo "Compiler: $lang $comp ($("$comp" --version 2> /dev/null | grep -m1 -iE 'version|release|[0-9]+\.[0-9]+'))"
  done
  grep -h '07.runtime.device ' "$R" | sed 's/^RESULT 07.runtime.device INFO /Device:   /'
  echo
  echo "Requirements"
  printf '  %-45s %s\n' "Virtual functions in device code (*)" "$(status '01\.|04\.image\.construct')"
  printf '  %-45s %s\n' "Device linking across static libraries (*)" "$(status '02\.')"
  printf '  %-45s %s\n' "Recursion, settable stack size" "$(status '03\.')"
  printf '  %-45s %s\n' "Single-source C++17, placement new" "$(status '01\.virtual\.same_kernel|04\.')"
  printf '  %-45s %s\n' "CUDA-like runtime API" "$(status '07\.')"
  printf '  %-45s %s\n' "Atomics on device memory" "$(status '06\.')"
  printf '  %-45s %s\n' "FP64" "$(status '05\.')"
  printf '  %-45s %s\n' "Build integration" "$(status '08\.')"
  echo "  (*) prerequisites"
  echo
  echo "Measurements"
  grep -h ' INFO ' "$R" | grep -v '07.runtime.device ' | sed 's/^RESULT /  /'
  echo
  echo "All results"
  sed 's/^RESULT /  /' "$R"
} > "$OUT/report.txt"
cat "$OUT/report.txt"
tar -czf "$OUT.tar.gz" "$OUT"
echo
echo "Please send back: $OUT.tar.gz"
