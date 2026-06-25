#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Self-contained, CMake/ninja-free per-config build of the gemm_harness ELF.
#
# Per-config inputs that change: ONLY the kernel (lowering_config in the .mlir).
# Constants reused as-is from build-rt (precompiled once, never rewritten here):
#   - harness.o            (build-rt/samples/gemm_square/CMakeFiles/gemm_harness.dir/harness.c.obj)
#   - libsnRuntime.a       (build-rt/snitch_cluster/libsnRuntime.a)
#   - the Quidditch dispatch lib + iree base/vm archives (build-rt/iree-configuration/...)
#   - base.ld              (resolved by the linker from the snitch_cluster -L path)
#   - the gemm_mod_module.c stub (#define EMITC_IMPLEMENTATION + #include "..._module.h")
#
# Each call builds in its OWN temp dir; nothing under build-rt is written, so
# concurrent builds cannot race. Extracted verbatim from:
#   build-rt/build.ninja:24582 (iree-compile custom command)
#   build-rt/build.ninja:24404 (.c -> bitcode .obj)
#   build-rt/build.ninja:24420 (llvm-ar/ranlib -> libgemm_mod.a)
#   build-rt/build.ninja:24489 + :1015 (final gemm_harness link)
import os
import re
import shutil
import subprocess
import tempfile

ROOT = "/home/dankeller/Projects/Quidditch"
BUILD = f"{ROOT}/build-rt"

# --- absolute toolchain paths (from `ninja -t commands gemm_harness`) ---
IREE_COMPILE = "/scratch/dankeller/snitch-compiler/iree-p6-build/tools/iree-compile"
XDSL_OPT = "/scratch/dankeller/snitch-compiler/Quidditch/.venv/bin/xdsl-opt"
TOOLCHAIN_ROOT = "/scratch/dankeller/snitch-compiler/Quidditch/toolchain"
LLVM = "/usr/scratch2/vulcano/colluca/tools/riscv32-snitch-llvm-almalinux8-15.0.0-snitch-0.5.0/bin"
CLANG = f"{LLVM}/clang"
LLVM_AR = f"{LLVM}/llvm-ar"
LLVM_RANLIB = f"{LLVM}/llvm-ranlib"
LD_LLD = f"{LLVM}/ld.lld"

# --- constant prebuilt inputs (reused, never regenerated) ---
HARNESS_O = f"{BUILD}/samples/gemm_square/CMakeFiles/gemm_harness.dir/harness.c.obj"
CONST_ARCHIVES = [
    f"{BUILD}/snitch_cluster/libsnRuntime.a",
    f"{BUILD}/iree-configuration/iree/runtime/plugins/Quidditch/src/Quidditch/dispatch/libQuidditch_dispatch_dispatch.a",
    f"{BUILD}/iree-configuration/iree/runtime/src/iree/vm/libiree_vm_impl.a",
    f"{BUILD}/iree-configuration/iree/runtime/src/iree/base/threading/libiree_base_threading_threading.a",
    f"{BUILD}/iree-configuration/iree/runtime/src/iree/base/internal/libiree_base_internal_memory.a",
    f"{BUILD}/iree-configuration/iree/runtime/src/iree/base/libiree_base_base.a",
    f"{BUILD}/iree-configuration/iree/runtime/src/iree/base/internal/libiree_base_internal_time.a",
]
# Linker -L search dirs (where base.ld lives + rtl libs); from build.ninja:24489.
LDIRS = [
    "/scratch/dankeller/snitch-compiler/Quidditch/snitch_cluster/sw/runtime",  # base.ld
    f"{ROOT}/runtime/snitch_cluster/rtl",
]
# Include dirs for the gemm_mod_module.c stub compile (build.ninja:24404). The
# only per-config-relevant one is the gemm_mod dir holding the generated .h; we
# point it at the per-config temp dir below. The rest are constant headers.
CONST_INCLUDES = [
    "/scratch/dankeller/snitch-compiler/Quidditch/iree",
    f"{BUILD}/iree-configuration/iree",
    "/scratch/dankeller/snitch-compiler/Quidditch/iree/third_party/printf/src",
    "/scratch/dankeller/snitch-compiler/Quidditch/iree/runtime/src",
    f"{BUILD}/iree-configuration/iree/runtime/src",
]
CONST_DEFS = [
    "-DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl",
    "-DIREE_PLATFORM_GENERIC",
    f'-DIREE_USER_CONFIG_H="{ROOT}/runtime/iree-configuration/config.h"',
    "-DPRINTF_SUPPORT_WRITEBACK_SPECIFIER=0",
    "-D_ISOC11_SOURCE",
]
# Common compile flags (build.ninja:600/748 LANGUAGE_COMPILE_FLAGS + std/lto).
CFLAGS = ["-mcpu=snitch", "-menable-experimental-extensions", "-mabi=ilp32d",
          "-mcmodel=medany", "-msmall-data-limit=0", "-O3", "-DNDEBUG",
          "-std=gnu11", "-flto=thin"]
# Link flags (build.ninja:1015 LINK_FLAGS) + the per-edge extras (build.ninja:24489).
LINK_FLAGS = ["-mcpu=snitch", "-menable-experimental-extensions", "-mabi=ilp32d",
              "-mcmodel=medany", "-msmall-data-limit=0", "-O3", "-DNDEBUG",
              "-flto=thin", "-mcpu=snitch", "-menable-experimental-extensions",
              "-mabi=ilp32d", "-mcmodel=medany", "-msmall-data-limit=0",
              f"-fuse-ld={LD_LLD}", "-nostartfiles", "-Wl,-mllvm,-target-abi=ilp32d"]

MLIR_TEMPLATE = f"{ROOT}/runtime/samples/gemm_square/gemm_square.mlir"

# Extra includes the harness.c needs beyond CONST_INCLUDES (from the ninja edge);
# HARNESS_O (the prebuilt gemm object) is defined above. Other ops compile their own.
HARNESS_INC = [
    f"{ROOT}/runtime/runtime/src",
    f"{BUILD}/iree-configuration/iree/runtime/plugins/Quidditch/src",
]
HARNESS_ISYSTEM = [
    "/scratch/dankeller/snitch-compiler/Quidditch/snitch_cluster/sw/runtime/api",
    "/scratch/dankeller/snitch-compiler/Quidditch/snitch_cluster/sw/deps/riscv-opcodes",
    f"{ROOT}/runtime/snitch_cluster/api",
]
# Minimal module header (the query declaration is config-invariant); lets the
# harness.o compile without iree-compile having run first.
_MODULE_H = """// generated by tools/autotune -- query declaration only
#include "iree/hal/local/executable_library.h"
#if __cplusplus
extern "C" {{
#endif
const iree_hal_executable_library_header_t** {query}(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment);
#if __cplusplus
}}
#endif
"""


def _inject(mlir_text, l1_tiles, dual_buffer, interchange=None):
    # N-dimensional: 3 tiles for matmul (M,N,K), 2 for a 2D elementwise op.
    tiles = ", ".join(str(t) for t in l1_tiles)
    s = re.sub(r"l1_tiles = \[[0-9, ]+\]", f"l1_tiles = [{tiles}]", mlir_text)
    s = re.sub(r"dual_buffer = (?:true|false)", f"dual_buffer = {dual_buffer}", s)
    if interchange is not None:
        ix = ", ".join(str(t) for t in interchange)
        s = re.sub(r"l1_tiles_interchange = \[[0-9, ]+\]", f"l1_tiles_interchange = [{ix}]", s)
    return s


def compile_harness(harness_c, module, query_symbol, out_obj, defines=()):
    """Compile a generated harness.c -> thin-LTO object (config-invariant; built
    once per op). Synthesizes a minimal <module>.h (query declaration) so it does
    not need iree-compile to have run first. `defines` adds extra -D flags (e.g.
    -DHARNESS_DUMP_OUTPUT for the Tier-2 cross-check). Returns (out_obj, None) or (None, err)."""
    tmp = tempfile.mkdtemp(prefix="qd_harness_")
    try:
        with open(os.path.join(tmp, f"{module}.h"), "w") as f:
            f.write(_MODULE_H.format(query=query_symbol))
        r = subprocess.run(
            [CLANG, *CONST_DEFS, *defines, f"-I{tmp}", *(f"-I{d}" for d in CONST_INCLUDES),
             *(f"-I{d}" for d in HARNESS_INC),
             *sum((["-isystem", d] for d in HARNESS_ISYSTEM), []),
             *CFLAGS, "-Wno-undefined-inline", "-o", out_obj, "-c", harness_c],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if r.returncode != 0:
            return None, f"clang(harness) failed:\n{r.stderr.strip()[-2000:]}"
        return out_obj, None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def build(config, outdir, mlir_template=None, module="gemm_mod", harness_obj=None,
          xdsl_passes=None):
    """Build a harness ELF for `config` into an isolated temp dir.

    config        = {"l1_tiles": [...], "dual_buffer": "true"|"false", "interchange": [...]}
                    l1_tiles/interchange are N-dim: 3 for matmul (M,N,K), 2 for a 2D op.
    outdir        = directory the final ELF is copied to (created if needed).
    mlir_template = kernel .mlir to inject into (defaults to gemm_square).
    module        = iree quidditch_module DST name (the static lib + header stem).
    harness_obj   = prebuilt harness .o to link (defaults to the gemm CMake object;
        other ops pass the output of compile_harness()).
    Returns (elf_path, None) on success or (None, error_string) if illegal/failed.
    The legality gate scans iree-compile stderr for 'error:' because
    --iree-quidditch-assert-compiled exits 0 even when it rejects a tiling.
    """
    os.makedirs(outdir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="qd_cfg_", dir=outdir)
    moddir = os.path.join(work, module)
    os.makedirs(moddir, exist_ok=True)

    # Per-config kernel: inject lowering_config into a temp copy of the .mlir.
    # Keep the original basename so iree-compile's embedded names match the ninja
    # build (the gemm canary asserts stripped-ELF identity).
    template = mlir_template or MLIR_TEMPLATE
    mlir = os.path.join(work, os.path.basename(template))
    with open(template) as f:
        base_mlir = f.read()
    with open(mlir, "w") as f:
        f.write(_inject(base_mlir, config["l1_tiles"], config["dual_buffer"],
                        config.get("interchange")))

    h = os.path.join(moddir, f"{module}_module.h")
    o = os.path.join(moddir, f"{module}.o")
    c = os.path.join(moddir, f"{module}_module.c")
    # The .c stub is constant; recreate it verbatim (build-rt's file(CONFIGURE)).
    with open(c, "w") as f:
        f.write(f'#define EMITC_IMPLEMENTATION\n#include "{module}_module.h"\n')

    # (a) iree-compile: regenerate the module from the per-config .mlir.
    r = subprocess.run(
        [IREE_COMPILE,
         "--iree-vm-bytecode-module-strip-source-map=true",
         "--iree-vm-emit-polyglot-zip=false",
         "--iree-input-type=auto",
         "--iree-input-demote-f64-to-f32=0",
         "--iree-hal-target-backends=quidditch",
         f"--iree-quidditch-static-library-output-path={o}",
         f"--iree-quidditch-xdsl-opt-path={XDSL_OPT}",
         f"--iree-quidditch-toolchain-root={TOOLCHAIN_ROOT}",
         "--iree-quidditch-assert-compiled=true",
         *([f"--iree-quidditch-xdsl-passes={xdsl_passes}"] if xdsl_passes else []),
         "--output-format=vm-c",
         "--iree-vm-target-index-bits=32",
         mlir, "-o", h],
        cwd=moddir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)
    if r.returncode != 0 or "error:" in r.stderr:
        shutil.rmtree(work, ignore_errors=True)
        return None, f"iree-compile rejected/failed:\n{r.stderr.strip()[-2000:]}"

    # (b) clang: compile the stub .c -> thin-LTO bitcode object.
    mod_obj = os.path.join(work, f"{module}_module.c.obj")
    r = subprocess.run(
        [CLANG, *CONST_DEFS, f"-I{moddir}", *(f"-I{d}" for d in CONST_INCLUDES),
         *CFLAGS, "-o", mod_obj, "-c", c],
        cwd=work, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)
    if r.returncode != 0:
        shutil.rmtree(work, ignore_errors=True)
        return None, f"clang(module) failed:\n{r.stderr.strip()[-2000:]}"

    # (c) llvm-ar + ranlib: pack <module>.o + the module obj into lib<module>.a.
    lib = os.path.join(work, f"lib{module}.a")
    subprocess.run([LLVM_AR, "cr", lib, o, mod_obj], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    subprocess.run([LLVM_RANLIB, lib], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # (d) final link: harness.o + lib<module>.a + the constant archives.
    elf = os.path.join(work, "harness")
    r = subprocess.run(
        [CLANG, *LINK_FLAGS, "-lm", "-Tbase.ld", harness_obj or HARNESS_O, "-o", elf,
         *(f"-L{d}" for d in LDIRS), lib, *CONST_ARCHIVES],
        cwd=work, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)
    if r.returncode != 0:
        shutil.rmtree(work, ignore_errors=True)
        return None, f"link failed:\n{r.stderr.strip()[-2000:]}"

    tag = "x".join(str(t) for t in config['l1_tiles']) + f"_db{config['dual_buffer']}"
    final = os.path.join(outdir, f"harness_{tag}")
    shutil.copy(elf, final)
    shutil.rmtree(work, ignore_errors=True)
    return final, None


if __name__ == "__main__":
    import sys
    cfg = {"l1_tiles": [16, 16, 16], "dual_buffer": "true"}
    out = sys.argv[1] if len(sys.argv) > 1 else "/scratch/dankeller/snitch-compiler/autotune-work/direct_out"
    elf, err = build(cfg, out)
    if err:
        print("BUILD-FAIL:", err)
        sys.exit(1)
    print("ELF:", elf)
