#!/usr/bin/env bats

setup() {
  KIT="$(cd "$(dirname "$BATS_TEST_FILENAME")/.." && pwd)"
  export KIT
}

@test "env.sh derives KIT_ROOT from its own location" {
  run bash -c "source '$KIT/env.sh' >/dev/null 2>&1; echo \$KIT_ROOT"
  [ "$status" -eq 0 ]
  [ "$output" = "$KIT" ]
}

@test "env.sh honours a pre-set KIT_DATA" {
  # `VAR=x source file` does NOT work here: a prefix assignment on a builtin is
  # unwound when it returns, so KIT_DATA would read back empty. Export it, which
  # is what a real caller does anyway.
  run bash -c "export KIT_DATA=/tmp/kitdata; source '$KIT/env.sh' >/dev/null 2>&1; echo \$KIT_DATA"
  [ "$output" = "/tmp/kitdata" ]
}

@test "env.sh derives every path from KIT_DATA" {
  run bash -c "export KIT_DATA=/tmp/kitdata; source '$KIT/env.sh' >/dev/null 2>&1; \
               echo \$MODEL; echo \$KIT_OUT"
  [[ "${lines[0]}" = "/tmp/kitdata/models/Qwen3-0.6B" ]]
  [[ "${lines[1]}" = "/tmp/kitdata/kit-out" ]]
}

@test "QUANT_DEVICE is cpu on a host with no GPU" {
  # tank has no GPU, so this must resolve to cpu without a torch import.
  run bash -c "source '$KIT/env.sh' >/dev/null 2>&1; echo \$QUANT_DEVICE"
  [[ "$output" = "cpu" || "$output" = "cuda" ]]
  if ! command -v nvidia-smi >/dev/null 2>&1; then [ "$output" = "cpu" ]; fi
}

@test "disk_guard passes when the requirement is trivially small" {
  # Assert disk_guard is a FUNCTION before calling it. Without this the test
  # passes vacuously with no env.sh at all: the source fails silently, the
  # disk_guard call is a 127 command-not-found, and the trailing echo still
  # exits 0.
  run bash -c "source '$KIT/env.sh' >/dev/null 2>&1; \
               [ \"\$(type -t disk_guard)\" = function ] || exit 3; \
               disk_guard 1 && echo OK"
  [ "$status" -eq 0 ]
  [[ "$output" == *OK* ]]
}

@test "disk_guard aborts when the requirement is absurdly large" {
  run bash -c "source '$KIT/env.sh' >/dev/null 2>&1; disk_guard 999999999"
  [ "$status" -ne 0 ]
  [[ "$output" == *"free space"* ]]
}

@test "disk_guard guards C: when /mnt/c exists, else KIT_DATA" {
  run bash -c "source '$KIT/env.sh' >/dev/null 2>&1; disk_guard_target"
  [ "$status" -eq 0 ]
  if [ -d /mnt/c ]; then
    [ "$output" = "/mnt/c" ]
  else
    [ "$output" = "$KIT_DATA" ]
  fi
}

# --grouped-gqa is structural, not a flag. There must be no way to express its
# absence anywhere in the kit -- in the old chain it was a positional
# pass-through on one script and a FUSE_FLAGS env var on two others, and
# omitting it on the latter silently shipped pre-fix attention in verify32 and
# the past-KV prefill while decode looked correct (6.836 tok/s, not 44.707).
@test "no FUSE_FLAGS escape hatch exists anywhere in the kit" {
  # Exclude this file: the comment above names the variable, and a self-match
  # would fail the test forever regardless of what the kit actually contains.
  run bash -c "grep -rn 'FUSE_FLAGS' '$KIT' | grep -v 'tests/test_env.bats'"
  [ "$status" -ne 0 ]
}

# The kit must run with the parent repo absent.
@test "kit does not reference the parent repo" {
  run bash -c "grep -rn 'LLMDEPLOY_ROOT\|LLMDEPLOY_DATA\|/mnt/x' '$KIT' \
                 --include='*.sh' --include='*.py' --include='*.bats' \
               | grep -v 'tests/test_env.bats'"
  [ "$status" -ne 0 ]
}
