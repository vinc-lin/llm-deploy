# Device test — ImageEncoder SIGSEGV bisect (Qwen3-VL-4B v2, SA8797P)

> **SUPERSEDED IN PART (v4).** The crash this document investigates is
> resolved: Genie's image staging interprets the input file as float32
> (nsp-image-model.cpp:501-524, `embedding-datatype` default), so the
> UFixed16 blobs v2/v3 shipped were read at 2x their size — a ~3 MB
> over-read, which is why padding and page-alignment probes could not
> help. v4 ships float32 blobs. Kept as the diagnostic record; do not
> run its probes. Current instructions: `V4_CHANGES.md`.

**Goal: two short runs that decide, between three possible causes, why
`node set image` dies — and capture the one tombstone line that lets us
disassemble the exact faulting instruction off-device.**

Context: your 2026-08-15 report (`QWEN3VL_E2E_DEPLOYMENT_STATUS.md` §11)
confirmed the v2 text tower fix — mask validates, both ctx-bins load, tokens
generate — and reported `SIGSEGV (SEGV_ACCERR)` at `GenieNode_setData+572`,
fault address exactly `base + 0x300000`, independent of the file's size or
content. This runbook is the follow-up experiment set.

Board time: **~15 minutes**. Nothing destructive — every new file lives in
`/data/local/tmp/qwen3vl` and can be deleted afterwards. The runs do **not**
load the 4B text tower (the crash fires at `node set image`, before
`pipeline execute`, so the ImageEncoder node alone reproduces it), which makes
each attempt a seconds-scale iteration instead of a multi-minute one.

## 0. TL;DR

| Run | Input file | Size (bytes) | Question it answers |
|---|---|---|---|
| **1** | `sample_image.raw` | 3,145,728 (exact) | Does a *correctly sized* blob crash at all? |
| **2** | `sample_pad.raw` (Run 1 blob + 4 KB zeros) | 3,149,824 | **Expected to fix it.** §1.2b resolves the mechanism off-device: a one-byte over-read into Scudo's guard page, on a heap buffer sized from *your* `fileSize`. Padding moves the guard page |
| every crash | — | — | Pull the tombstone. Frame `#00`'s raw pc offset is the single most valuable byte of data on this bug. |

Then report per §7. Decision tree in §6.

## 1. What we know — and what `+572` actually is

### 1.1 The bug class (from the SDK's own qualla source)

`nsp-image-model.cpp:499–523` (`setupInput`, the UFixed16 path our ViT takes):

```cpp
size_t numElements = std::accumulate(input_tensor->tensor->v1.dimensions,
                      input_tensor->tensor->v1.dimensions + rank, 1, multiplies);
size_t bufferSize = d_inputs[name].bw() * numElements;   // from the GRAPH, not the caller
...
std::copy(inputs.data(), inputs.data() + bufferSize,     // never checks inputs.size()
          reinterpret_cast<uint8_t*>(getBuffer(input_tensor)));
```

The copy length comes from the graph's tensor spec and is **never compared to
the caller's buffer size**. For our ViT, `pixel_values` is `[1024,1536]`
`UFIXED_POINT_16` → 2 × 1,572,864 = **3,145,728 = 0x300000** bytes — exactly
the fault offset you observed, and exactly the size of every shipped `.raw`
(verified, all seven, §2). On the caller side, `genie-app` allocates precisely
`new int8_t[fileSize]` and validates only `fileSize > 0`
(`genie-app/main.cpp:1162–1184`) — no exact-size check anywhere.

This **fully explains the 33-byte crash**: `setData` reads 3 MB out of a
33-byte allocation, whatever the file contains. It does **not** explain a crash
with a correctly sized input — a 0x300000-byte copy from a 0x300000-byte buffer
is exactly in-bounds. Something extra is being touched, on one side or the
other. Which side is the whole question.

### 1.2 Caveat, learned the hard way this week

**The SDK's `examples/Genie` source tree is not the source of the shipped
`libGenie.so`.** Your board proved it: the binary rejects unknown `QnnHtp`
config keys, and no such validation exists anywhere in the example source (the
decode-only fallback we shipped was derived from that source — that failure is
on us; the fallback is being removed from the bundle). So treat §1.1 as the
*bug class*, not gospel: the shipped copy loop may differ by exactly the
off-by-one this experiment is hunting.

### 1.2b What the binary actually does (resolved off-device, 2026-08-16)

We walked the vtable graph in the byte-identical `libGenie.so` through its RELA
relocations and disassembled the real callee. Three findings change the picture:

**1. The buffer is not a DMA buffer — it is an ordinary heap allocation, sized
by the caller.** `genie::pipeline::ImageEncoder`'s vtable slot `+0x90` resolves
to `0x3ef514`, and that function does:

```
+436   tbnz  x20, #0x3f, ...      ; reject negative size
+440   mov   x0, x20              ; x20 = the size YOU passed
+444   bl    operator new         ; allocate exactly that many bytes
+448   mov   x1, x21              ; your data pointer
+452   mov   x2, x20
+460   bl    memcpy               ; copy exactly that many bytes
+512   add   x20, x22, x20        ; end = begin + size
+540   stp   x22, x20, [x23,#0x28] ; std::vector {begin, end, cap}
```

It null-checks your pointer and zero-checks your size first. So the node makes
its **own** `std::vector` of exactly `fileSize` bytes. The graph-derived length
is applied later, downstream, against *that* vector.

**2. The graph demands exactly the size we ship.** The ViT ctx-bin declares
`pixel_values [1024,1536] UFIXED_POINT_16` = 1,572,864 × 2 = **3,145,728 bytes
= 0x300000**, and all seven shipped `.raw` blobs are exactly 3,145,728 bytes.
So the downstream copy is *exactly* in bounds — its last byte is
`base + 0x2FFFFF`.

**3. `SEGV_ACCERR` at exactly `base + 0x300000` is a guard-page signature.**
`SEGV_ACCERR` is a *permission* fault, not a missing mapping (`SEGV_MAPERR`).
Android's Scudo allocator services a 3 MB request from its secondary allocator,
which maps the region page-aligned with a **PROT_NONE guard page immediately
after** — and 3,145,728 is exactly 768 pages, so the guard begins at precisely
`base + 0x300000`. A read of one byte past the end lands on it and raises
exactly the fault you saw, at exactly the address you saw.

**Conclusion: this is an off-by-one over-read of the source buffer (H-src),
not a DMA/destination problem.** Something forms and dereferences the
end pointer — a `<=` where `<` was meant, or a length of `bufferSize + 1`.

**And that makes padding a real fix, not just a probe.** Because the allocation
is sized from *your* `fileSize`, adding bytes to the file enlarges the heap
allocation and pushes the guard page beyond the over-read, while the graph still
consumes only the first 0x300000 bytes. The padding never reaches the tensor.

Run 2 below is therefore expected to **succeed**. If it does, padded blobs are a
legitimate shipping workaround and we re-cut the bundle the same day.

### 1.3 `GenieNode_setData+572` is a call site, not the faulting access

We disassembled the byte-identical `libGenie.so` you MD5-matched
(`c57ddd9161091ae8117a55ca6c4f16d1`). `GenieNode_setData` sits at `0x2cf248`,
2,244 bytes, and `+572` is:

```
  +544   ldr   x8, [x24]          ; load object vtable
  +548   ldr   x8, [x8, #0x90]    ; vtable slot 0x90 = node-type-specific setData
  +552   add   x4, sp, #0x68
  +556   mov   x0, x24
  +560   mov   w1, w20
  +564   mov   x2, x19            ; caller's data pointer
  +568   mov   x3, x21            ; caller's data size
  +572   blr   x8                 ; <=== the reported "crash location"
```

`blr x8` is an indirect **call** — it touches no data memory. The tombstone
frame that reads `GenieNode_setData+572` is the *return address* of the frame
that called into the real per-node-type `setData` implementation (an internal,
stripped symbol). **The faulting instruction is in that callee**, which is why
§5 asks for the raw `#00 pc` offset: with it, we disassemble the exact
instruction host-side and read off whether it is a load (source overread) or a
store (destination overrun), and from which register — no symbols needed.

### 1.4 The three hypotheses

- **H-untested** — the exact-size real blobs were never run in isolation, and
  only the deliberately mis-sized experiments crashed (all of which §1.1
  explains). Run 1 kills or confirms this in one shot.
- **H-src** — the shipped copy reads ≥1 byte past the **source** heap buffer.
  Consistent with `SEGV_ACCERR` at `base + 0x300000`: a 3 MB `new[]` goes to
  the allocator's large-allocation path, which maps it page-aligned and places
  an inaccessible guard page immediately after — and 3,145,728 is exactly 768
  pages, so a one-byte overread lands precisely on the guard. Padding the file
  (Run 2) makes this crash vanish, and it is then a legal shipping workaround.
- **H-dst** — the callee touches one byte past the **destination** DMA/rpcmem
  mapping (also `SEGV_ACCERR` at its `base + 0x300000`). No host-side change
  can fix that; §6 branch D is the Qualcomm escalation package.

One clue cuts against H-src: you observed the same `+0x300000` arithmetic
*independent of file size*. Under H-src the source base and surrounding layout
change with allocation size (a 33-byte file is a small-size-class chunk inside
a larger mapped region — an overread there faults at a much less predictable
address, or not at all). If the fault address was literally stable across
sizes, that leans H-dst. Frame `#00` settles it either way.

## 2. Pre-flight (2 min)

All commands run in the board shell (`adb shell`, or however you normally
reach it), from the bundle directory:

```sh
cd /data/local/tmp/qwen3vl
```

Confirm the blobs are the exact-size originals (any prior experiment may have
overwritten them):

```sh
ls -l sample_image.raw wx_*.raw          # every one must be exactly 3145728
md5sum sample_image.raw libGenie.so
```

Expected:

```
f69a55393eedf6f4b0011dd02c7cbcc4  sample_image.raw
c57ddd9161091ae8117a55ca6c4f16d1  libGenie.so
```

If `sample_image.raw` mismatches, re-pull it from the bundle before Run 1 —
the whole experiment hinges on its size being exactly 3,145,728.

Clear the crash logs so each run's forensics are unambiguous:

```sh
logcat -c
ls /data/tombstones/          # note what already exists; new files = this run
```

## 3. Run 1 — exact-size baseline

Create the minimal script (ImageEncoder node only — no LUT, no text tower):

```sh
cat > imgonly.script <<'EOF'
version
node config create imageEncoderConfig genie_image_encoder_qwen3vl.json
node create imageEncoder imageEncoderConfig
node set image imageEncoder GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT sample_image.raw
node free imageEncoder
EOF
```

Run it:

```sh
LD_LIBRARY_PATH=. ./genie-app -s imgonly.script 2>&1 | tee run1.log
echo "exit=$?"
```

(If `genie-app` refuses `node set` outside a pipeline — we do not expect it
to — fall back to the full `wx_clear.script`, which crashes at the same line.)

**Outcomes:**

- **No crash, exit 0** → H-untested confirmed: correctly sized blobs are fine
  and the earlier signature came from mis-sized experiment files, exactly as
  §1.1 predicts. Skip Run 2, go straight to §6 branch A (run the full
  pipeline — you may be one command away from the first on-device caption).
- **SIGSEGV** → capture forensics (§5), then Run 2.
- **Clean error message, no crash** → record it verbatim (§7); the shipped
  binary is doing something the example source does not, and the message will
  say what.

## 4. Run 2 — padded blob (the discriminator)

Only if Run 1 crashed. Append one page of zeros to a *copy* (never modify the
originals):

```sh
cp sample_image.raw sample_pad.raw
dd if=/dev/zero bs=4096 count=1 >> sample_pad.raw
ls -l sample_pad.raw                       # must be 3149824
sed 's/sample_image\.raw/sample_pad.raw/' imgonly.script > imgpad.script
LD_LIBRARY_PATH=. ./genie-app -s imgpad.script 2>&1 | tee run2.log
echo "exit=$?"
```

Why padding is *safe*, not a hack: `genie-app` reads the whole file and
allocates exactly `fileSize` bytes; `setData` copies only the tensor-derived
0x300000 bytes into the graph. The trailing zeros never reach the HTP — they
exist purely to give the source allocation slack beyond the copy's end.

**Outcomes:**

- **No crash** → **H-src confirmed.** The shipped `libGenie` overreads the
  source buffer, the padding absorbs it, and this is a shipping workaround: we
  re-cut the bundle with padded blobs the same day (§6 branch C).
- **Same SIGSEGV** → **H-dst.** The fault is past the destination mapping and
  no host-side change can reach it (§6 branch D). Capture forensics for *this*
  run too — comparing the two tombstones' fault addresses (same address ⇒
  destination; shifted by the padding ⇒ something stranger) is itself evidence.
- **Clean error message** (e.g. a size validation) → record verbatim; it
  proves the binary checks sizes and tells us the expected value.

## 5. Forensics — on every crashing run

```sh
ls -t /data/tombstones/ | head -3          # newest = this crash
cp /data/tombstones/<newest> /data/local/tmp/qwen3vl/tombstone_runN.txt
logcat -d > /data/local/tmp/qwen3vl/logcat_runN.txt
```

Pull both files off the board. From the tombstone we specifically need:

1. **The signal line** — `signal 11 (SIGSEGV), code 2 (SEGV_ACCERR), fault
   addr 0x…` — the exact fault address.
2. **The backtrace, frames `#00`–`#04`, with raw pc offsets** — e.g.
   `#00 pc 00000000005a1b2c  /data/local/tmp/qwen3vl/libGenie.so`. Frame
   `#00`'s offset is the prize: we hold the byte-identical `.so`, so that one
   number gives us the faulting instruction, its direction (load = source /
   store = destination), and which pointer register it dereferenced. It
   decides H-src vs H-dst even if Run 2's result is ambiguous.
3. **The register dump** (`x0…x30, sp, pc`) — the registers around the
   faulting access, interpreted against the disassembly.
4. **The memory-map excerpt around the fault address** — modern tombstones
   mark it (`--->Fault address falls at …` or the map rows flanking it). This
   tells us *what* the guard region is: an allocator guard page after a heap
   block (H-src) or the end of a `/dev/dma_heap` / ION / kgsl mapping (H-dst).

If tombstones are disabled in this GVM image, say so — the `logcat -d` crash
dump usually still carries the signal line and backtrace, which covers items
1–2.

## 6. Decision tree

> **Status as of bundle v3 (2026-08-17): the fix in branch C below has already
> been applied to the shipped bundle.** Every `.raw` in v3 is payload + 4096
> zero bytes (3,149,824 B, was 3,145,728), generated that way by
> `preprocess_image.py` / `build_test_kit.py` and enforced by
> `lint_pipeline_bundle.py`. So on v3 you do **not** start from Run 1 — you
> just run the pipeline. This section is now the *diagnostic to reach for only
> if v3 still crashes on `node set image`*, and in that case Run 1's
> exact-size blob is recreated from the padded one with
> `head -c 3145728 sample_image.raw > nopad.raw` rather than shipped
> separately.

| | Run 1 (unpadded) | Run 2 (padded) | Meaning | Next action |
|---|---|---|---|---|
| **A** | no crash | — | Exact-size blobs were never the problem (H-untested) | Run the full `genie_pipeline_qwen3vl.script`, then the `wx_*.script` kit — captions expected. Report per the bundle README's test plan |
| **B** | error msg | — | Shipped binary validates input size (diverges from example source) | Send the message verbatim; we resize/repack to what it demands |
| **C** | crash | no crash | **H-src**: source overread, guard page — **confirmed hypothesis** | Nothing to re-cut: v3 already ships padded blobs and every `wx_*.script` already points at one. Continue straight into the kit and the test plan |
| **D** | crash | crash | **H-dst**: overrun past the DMA/rpcmem destination | Not host-fixable. We assemble the Qualcomm escalation package (both tombstones, `imgonly.script`, blob MD5s, ViT `info.json`, QAIRT/Genie versions). Text-only deployment continues in parallel — and on v3 that path is in-bundle (`genie-t2t-run` + `genie_dialog_qwen3vl_4b.json`) |

In every branch, the text-only performance test (`genie-t2t-run`, per your
report "ready to run") is **unblocked and worth doing now** — the first tok/s
numbers for a 4B W8A16 tower on this silicon stand on their own.

## 7. Report back

Per run: crash yes/no, `exit=` value, the tail of `runN.log`, and for crashes
the two pulled files (`tombstone_runN.txt`, `logcat_runN.txt`). A filled row
set of the §6 table plus attachments is a complete report — no prose needed.

If you have appetite for one optional extra: re-run the 33-byte-file case
under `imgonly.script` and pull *its* tombstone. If its frame `#00` offset
differs from Run 1's, the mis-sized and exact-size crashes are two different
faults — worth knowing before anyone pads anything.
