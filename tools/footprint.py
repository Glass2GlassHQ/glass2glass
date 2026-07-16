#!/usr/bin/env python3
"""Worst-case RAM / stack / ROM report for a bare-metal g2g pipeline ELF (M627).

Reads a fully linked (gc-sectioned) ELF and reports the numbers an MCU
integrator budgets against:

  ROM         = loadable code + constants (.text, .rodata, .ARM.exidx, .data)
  static RAM  = .data + .bss (the no-alloc pipeline keeps this at 0: all state
                lives in the entry function's stack frame)
  stack       = worst-case call-chain stack depth from the entry symbol,
                computed from the disassembly (frame = pushes + sp-decrements),
                so the number covers the pipeline state machine + capture ring,
                which are locals of the entry function

The stack walk is a whole-call-graph DFS with cycle detection. An indirect
call (blx <reg>, here only the noop-waker vtable) is bounded conservatively by
the deepest chain of any non-entry function in the ELF. Any sp-modifying
instruction the parser does not model fails the run rather than under-report.

Usage:
  footprint.py ELF --entry SYM [--max-rom N] [--max-stack N] [--max-static-ram N]

Exit nonzero if a budget is exceeded, recursion is found, or an unmodeled
sp write appears. Requires llvm-objdump + llvm-size: pass their paths with
--objdump / --size (footprint-report.sh resolves them from the rustup
llvm-tools component when they are not on PATH; a GNU objdump built for the
host cannot disassemble the ARM ELF).
"""

import argparse
import re
import shutil
import subprocess
import sys

# ISA-agnostic: sections absent on a given target contribute 0, so one set
# covers ARM (.ARM.exidx) and RISC-V (.srodata / .sdata / .sbss small-data).
ROM_SECTIONS = {".text", ".rodata", ".ARM.exidx", ".data", ".srodata", ".sdata"}
RAM_SECTIONS = {".data", ".bss", ".sdata", ".sbss"}


def section_sizes(size_tool, elf):
    out = subprocess.run(
        [size_tool, "-A", elf], capture_output=True, text=True, check=True
    ).stdout
    sizes = {}
    for line in out.splitlines():
        m = re.match(r"^(\.\S+)\s+(\d+)\s+\d+", line)
        if m:
            sizes[m.group(1)] = int(m.group(2))
    return sizes


def reg_count(braces):
    # "{r4, r5, r6, r7, lr}" or "{r4-r7, lr}" -> number of registers.
    n = 0
    for part in braces.strip("{} ").split(","):
        part = part.strip()
        m = re.match(r"^[rd](\d+)\s*-\s*[rd](\d+)$", part)
        n += (int(m.group(2)) - int(m.group(1)) + 1) if m else 1
    return n


def frame_and_calls_arm(insn, name, funcs, frame, callees, regs):
    """ARM/Thumb-2 stack-frame + call model. (`regs` is unused: these builds
    encode every frame as an sp immediate, so no register tracking is needed.)"""
    if m := re.match(r"^push(?:\.w)?\s+(\{.*\})", insn):
        funcs[name][0] = frame + 4 * reg_count(m.group(1))
    elif m := re.match(r"^vpush(?:\.w)?\s+(\{.*\})", insn):
        funcs[name][0] = frame + 8 * reg_count(m.group(1))
    elif m := re.match(r"^stmdb(?:\.w)?\s+sp!,\s*(\{.*\})", insn):
        funcs[name][0] = frame + 4 * reg_count(m.group(1))
    elif m := re.match(r"^subs?(?:\.w)?\s+sp,\s*(?:sp,\s*)?#(0x[0-9a-f]+|\d+)", insn):
        funcs[name][0] = frame + int(m.group(1), 0)
    elif m := re.match(r"^subw\s+sp,\s*sp,\s*#(0x[0-9a-f]+|\d+)", insn):
        funcs[name][0] = frame + int(m.group(1), 0)
    elif m := re.match(r"^str\w*\s+\S+,\s*\[sp,\s*#-(0x[0-9a-f]+|\d+)\]!", insn):
        funcs[name][0] = frame + int(m.group(1), 0)
    elif re.match(r"^add\w*\s+sp,\s*(?:sp,\s*)?#", insn):
        pass  # immediate sp increase = frame release (epilogue), never growth
    elif re.match(r"^(sub|subs|subw|mov|movs|add\w*)\s+sp\b", insn):
        sys.exit(f"FAIL: unmodeled sp write in {name}: {insn}")
    # Call edges: bl / tail-branch to another symbol; blx <reg> is indirect.
    if m := re.match(r"^(?:bl|b|b\.w)\s+0x[0-9a-f]+\s+<([^+>]+)>", insn):
        if m.group(1) != name:
            callees.add(m.group(1))
    elif re.match(r"^blx\s+r\d+", insn):
        funcs[name][2] = True


# RISC-V instructions whose first operand is NOT an integer destination
# register (stores write memory, branches / plain jumps / system ops write no
# GPR). The ISA is otherwise regular: rd is the first operand for every
# register-writing op, so anything not on this list defines its first operand.
_RV_NON_WRITING = frozenset((
    "sb", "sh", "sw", "sd", "fsb", "fsh", "fsw", "fsd", "fsq",
    "beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez",
    "blez", "bgez", "bltz", "bgtz", "bgt", "ble", "bgtu", "bleu",
    "j", "jr", "ret", "tail", "fence", "ecall", "ebreak", "nop",
    "swsp", "sdsp", "fswsp", "fsdsp", "unimp",
))
_RV_REG = re.compile(r"^(?:x\d+|zero|ra|sp|gp|tp|fp|[atsx]\d+)$")


def _rv_reg_value(tok, regs):
    """The tracked constant in `tok`, or None if not statically known. x0/zero
    always reads 0."""
    tok = tok.strip()
    if tok in ("zero", "x0"):
        return 0
    return regs.get(tok)


def _rv_dest(insn, base):
    """The integer destination register `insn` writes, or None if it writes no
    GPR (store / branch / jump-without-link / system)."""
    if base in _RV_NON_WRITING:
        return None
    parts = insn.split(None, 1)
    if len(parts) < 2:
        return None
    first = parts[1].split(",")[0].strip()
    return first if _RV_REG.match(first) else None


def _rv_const_result(insn, base, regs):
    """Value the instruction leaves in its destination when it is a constant-
    materialization op with statically known inputs, else None (the caller then
    invalidates the destination). Masked to 32 bits, matching rv32 wrap."""
    if base == "lui":
        if m := re.match(r"^\S+\s+\w+,\s*(0x[0-9a-f]+|\d+)", insn):
            return (int(m.group(1), 0) << 12) & 0xFFFFFFFF
    elif base == "li":
        if m := re.match(r"^\S+\s+\w+,\s*(-?(?:0x[0-9a-f]+|\d+))", insn):
            return int(m.group(1), 0) & 0xFFFFFFFF
    elif base == "mv":
        if m := re.match(r"^\S+\s+\w+,\s*(\w+)", insn):
            return _rv_reg_value(m.group(1), regs)
    elif base == "addi":
        if m := re.match(r"^\S+\s+\w+,\s*(\w+),\s*(-?(?:0x[0-9a-f]+|\d+))", insn):
            b = _rv_reg_value(m.group(1), regs)
            return (b + int(m.group(2), 0)) & 0xFFFFFFFF if b is not None else None
    elif base == "slli":
        if m := re.match(r"^\S+\s+\w+,\s*(\w+),\s*(0x[0-9a-f]+|\d+)", insn):
            b = _rv_reg_value(m.group(1), regs)
            return (b << int(m.group(2), 0)) & 0xFFFFFFFF if b is not None else None
    elif base == "add":
        if m := re.match(r"^\S+\s+\w+,\s*(\w+),\s*(\w+)", insn):
            a = _rv_reg_value(m.group(1), regs)
            c = _rv_reg_value(m.group(2), regs)
            return (a + c) & 0xFFFFFFFF if a is not None and c is not None else None
    return None


def frame_and_calls_riscv(insn, name, funcs, frame, callees, regs):
    """RISC-V (rv32) stack-frame + call model. The frame is either a single
    `addi sp, sp, -N` or, when N exceeds the 12-bit `addi` immediate, a constant
    materialized into a register (`lui`/`addi`/`slli`/...) then `sub sp, sp,
    <reg>`; that constant is resolved from the per-function `regs` map, so a
    large fixed frame is reported exactly rather than refused. Register saves
    are `sw <reg>, off(sp)` (sp as base, not destination, so they are not frame
    growth). Direct calls are `auipc`+`jalr`, and llvm-objdump annotates the
    target symbol on the `jalr` line; an indirect call/jump is a `jalr`/`jr`
    through a register with no symbol."""
    base = insn.split()[0] if insn else ""
    if base.startswith("c."):
        base = base[2:]  # compressed alias shares the base mnemonic's model
    # Frame allocation only when sp is the destination (`addi sp, sp, -N`).
    # `addi a3, sp, N` (sp as source, computing an address) must not match.
    if m := re.match(r"^(?:c\.)?addi\s+sp,\s*sp,\s*-(0x[0-9a-f]+|\d+)", insn):
        funcs[name][0] = frame + int(m.group(1), 0)
    elif re.match(r"^(?:c\.)?addi\s+sp,\s*sp,\s*(0x[0-9a-f]+|\d+)\b", insn):
        pass  # positive immediate = frame release (epilogue)
    elif m := re.match(r"^sub\s+sp,\s*sp,\s*(\w+)", insn):
        # Register-materialized frame: a fixed size too large for addi's 12-bit
        # immediate, so rustc builds the constant in a register first. Resolve
        # it; refuse (fail) rather than under-report if it is not a known const.
        val = _rv_reg_value(m.group(1), regs)
        if val is None:
            sys.exit(f"FAIL: unresolved sub sp,sp,{m.group(1)} in {name} "
                     f"(register is not a tracked compile-time constant)")
        funcs[name][0] = frame + val
    elif re.match(r"^add\s+sp,\s*sp,\s*\w+", insn):
        pass  # register-materialized frame release (the sub-sp epilogue pair)
    elif re.match(r"^(?:c\.)?(add|addi|sub|mv|and|andi|or|ori|sll|slli|xor)\w*\s+sp\b", insn):
        # Any other instruction writing sp (as destination) is unmodeled.
        sys.exit(f"FAIL: unmodeled sp write in {name}: {insn}")
    # Track compile-time constants in GPRs so the `sub sp, sp, <reg>` above can
    # be resolved. A modeled op with known inputs sets the value; any other
    # write to a register drops it (so a stale constant can never be reused).
    dest = _rv_dest(insn, base)
    if dest is not None and dest not in ("sp", "zero", "x0"):
        val = _rv_const_result(insn, base, regs)
        if val is None:
            regs.pop(dest, None)
        else:
            regs[dest] = val
    # Call / jump edges. A trailing <symbol> on a jal/jalr is a resolved direct
    # call target; a symbol-less jalr/jr through a register is indirect (the
    # noop-waker vtable poll, a jump table). `ret` is a return, not an edge.
    op = insn.split()[0] if insn else ""
    sym = re.search(r"<([^+>]+)>\s*$", insn)
    if op in ("jal", "jalr", "j") and sym:
        if sym.group(1) != name:
            callees.add(sym.group(1))
    elif op in ("jalr", "jr", "c.jalr", "c.jr") and not sym:
        funcs[name][2] = True


def parse_functions(objdump_tool, elf, isa):
    """-> {name: (frame_bytes, set(callee_names), has_indirect_call)}"""
    out = subprocess.run(
        [objdump_tool, "-d", "--no-show-raw-insn", elf],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    model = frame_and_calls_riscv if isa == "riscv" else frame_and_calls_arm
    funcs, name, regs = {}, None, {}
    for line in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
        if m:
            name = m.group(1)
            funcs[name] = [0, set(), False]
            regs = {}  # constant-tracking state is per function
            continue
        if name is None or "\t" not in line:
            continue
        insn = line.split("\t", 1)[1].strip()
        frame, callees, _ = funcs[name]
        model(insn, name, funcs, frame, callees, regs)
    return {k: (v[0], v[1], v[2]) for k, v in funcs.items()}


def max_depth(funcs, name, memo, path, indirect_bound):
    if name not in funcs:
        return 0  # e.g. a PLT-less libc symbol; bare-metal ELFs resolve all
    if name in path:
        sys.exit(f"FAIL: recursion through {name}; worst-case stack is unbounded")
    if name in memo:
        return memo[name]
    frame, callees, indirect = funcs[name]
    deepest = indirect_bound if indirect else 0
    path.add(name)
    for c in callees:
        deepest = max(deepest, max_depth(funcs, c, memo, path, indirect_bound))
    path.discard(name)
    memo[name] = frame + deepest
    return memo[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("elf")
    ap.add_argument("--entry", required=True)
    ap.add_argument("--max-rom", type=int)
    ap.add_argument("--max-stack", type=int)
    ap.add_argument("--max-static-ram", type=int)
    ap.add_argument("--objdump", default=shutil.which("llvm-objdump"))
    ap.add_argument("--size", default=shutil.which("llvm-size"))
    ap.add_argument("--isa", choices=("arm", "riscv"), default="arm")
    args = ap.parse_args()
    if not args.objdump or not args.size:
        sys.exit("FAIL: llvm-objdump / llvm-size not found; pass --objdump / --size")

    sizes = section_sizes(args.size, args.elf)
    rom = sum(sizes.get(s, 0) for s in ROM_SECTIONS)
    ram = sum(sizes.get(s, 0) for s in RAM_SECTIONS)

    funcs = parse_functions(args.objdump, args.elf, args.isa)
    if args.entry not in funcs:
        sys.exit(f"FAIL: entry symbol {args.entry} not in ELF")
    # Conservative stand-in for the one indirect call (the noop-waker vtable,
    # whose real targets are empty): the deepest chain of any non-entry
    # function, so the bound holds whatever the register points at.
    indirect_bound = 0
    for name in funcs:
        if name != args.entry and not funcs[name][2]:
            indirect_bound = max(indirect_bound, max_depth(funcs, name, {}, set(), 0))
    stack = max_depth(funcs, args.entry, {}, set(), indirect_bound)

    print(f"ROM (.text+.rodata+.ARM.exidx+.data): {rom} bytes")
    print(f"static RAM (.data+.bss):              {ram} bytes")
    print(f"worst-case stack from {args.entry}: {stack} bytes")
    print(f"  entry frame {funcs[args.entry][0]} bytes "
          f"(holds the capture ring + pipeline state machine)")
    print(f"  indirect-call bound {indirect_bound} bytes")

    failed = False
    for label, value, budget in (
        ("ROM", rom, args.max_rom),
        ("stack", stack, args.max_stack),
        ("static RAM", ram, args.max_static_ram),
    ):
        if budget is not None and value > budget:
            print(f"FAIL: {label} {value} exceeds budget {budget}")
            failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
