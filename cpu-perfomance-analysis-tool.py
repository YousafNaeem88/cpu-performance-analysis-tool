

import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
# 1. CPU MODEL  (Registers, Memory, Cache, ALU)
# ─────────────────────────────────────────────

# Registers
registers = {'AX': 0, 'BX': 0, 'CX': 0, 'DX': 0}

# Memory: 256 bytes, pre-loaded with some values for demo
memory = [0] * 256
for i in range(256):
    memory[i] = i * 2          # simple pattern so cache misses are interesting

# Cache: direct-mapped, max 16 entries  {address: value}
CACHE_SIZE = 16
cache = {}

# Performance counters
execution_time  = 0            # total cycles
cache_hits      = 0
cache_misses    = 0
branch_correct  = 0
branch_wrong    = 0

# Per-instruction cycle log (for the bar chart)
cycle_log = []                 # [(instruction_label, cycles_used)]


# ─────────────────────────────────────────────
# ALU: Arithmetic / Logic Unit
# ─────────────────────────────────────────────

def ALU(op, a, b):
    """Perform a basic arithmetic or logical operation."""
    ops = {
        'ADD': lambda x, y: x + y,
        'SUB': lambda x, y: x - y,
        'MUL': lambda x, y: x * y,
        'AND': lambda x, y: x & y,
        'OR' : lambda x, y: x | y,
        'XOR': lambda x, y: x ^ y,
    }
    if op in ops:
        return ops[op](a, b)
    raise ValueError(f"Unknown ALU op: {op}")


# ─────────────────────────────────────────────
# Cache helper
# ─────────────────────────────────────────────

def load_from_memory(address):
    """
    Load a value from memory through the cache.
    Direct-mapped cache: key = address % CACHE_SIZE.
    Updates cache_hits / cache_misses and execution_time.
    Returns the value at `address`.
    """
    global cache_hits, cache_misses, execution_time

    if address in cache:
        cache_hits += 1
        execution_time += 1          # cache hit  → 1 cycle
    else:
        cache_misses += 1
        execution_time += 10         # cache miss → 10 cycles (penalty)
        # evict oldest entry if cache is full
        if len(cache) >= CACHE_SIZE:
            oldest_key = next(iter(cache))
            del cache[oldest_key]
        cache[address] = memory[address]

    return cache[address]


# ─────────────────────────────────────────────
# 2. FETCH → DECODE → EXECUTE CYCLE
# ─────────────────────────────────────────────

def execute_instruction(instr):
    """
    Fetch  – pick the instruction (done by iterating the list).
    Decode – read the opcode and operands.
    Execute– carry out the operation and update counters.
    """
    global execution_time, branch_correct, branch_wrong

    op = instr[0]

    # ── MOV  reg ← immediate value ──────────────────────────
    if op == 'MOV':
        _, reg, val = instr
        registers[reg] = val
        execution_time += 1
        cycle_log.append((f"MOV {reg},{val}", 1))

    # ── LOAD reg ← memory[address] ──────────────────────────
    elif op == 'LOAD':
        _, reg, address = instr
        registers[reg] = load_from_memory(address)
        cycles = 1 if address in cache else 10    # already updated inside fn
        cycle_log.append((f"LOAD {reg},[{address}]", cycles))

    # ── ALU ops: ADD, SUB, MUL, AND, OR, XOR ────────────────
    elif op in ('ADD', 'SUB', 'MUL', 'AND', 'OR', 'XOR'):
        _, dst, src = instr
        # src can be a register name or a literal integer
        src_val = registers[src] if isinstance(src, str) else src
        registers[dst] = ALU(op, registers[dst], src_val)
        execution_time += 1
        cycle_log.append((f"{op} {dst},{src}", 1))

    # ── CMP  set a flag register (stored in DX for simplicity) ──
    elif op == 'CMP':
        _, reg, val = instr
        registers['DX'] = 1 if registers[reg] == val else 0
        execution_time += 1
        cycle_log.append(("CMP", 1))

    # ── JMP  unconditional jump (simulated – we just note it) ───
    elif op == 'JMP':
        execution_time += 2
        cycle_log.append(("JMP", 2))

    # ── JEQ  jump-if-equal  (branch prediction demo) ────────────
    elif op == 'JEQ':
        _, prediction = instr          # 1 = predict-taken, 0 = predict-not-taken
        actual = registers['DX']       # result of last CMP
        if prediction == actual:
            branch_correct += 1
        else:
            branch_wrong += 1
        execution_time += 3 if prediction != actual else 1   # misprediction penalty
        cycle_log.append(("JEQ", 3 if prediction != actual else 1))

    else:
        print(f"  [WARNING] Unknown opcode: {op}")


# ─────────────────────────────────────────────
# 3. INSTRUCTION PROGRAM
# ─────────────────────────────────────────────

instructions = [
    # Basic moves and arithmetic
    ('MOV',  'AX', 10),
    ('MOV',  'BX', 20),
    ('ADD',  'AX', 'BX'),          # AX = 10 + 20 = 30
    ('SUB',  'BX', 5),             # BX = 20 - 5 = 15
    ('MUL',  'AX', 2),             # AX = 30 * 2 = 60

    # Memory loads (mix of hits and misses)
    ('LOAD', 'CX', 0),             # miss  – cold cache
    ('LOAD', 'CX', 4),             # miss
    ('LOAD', 'CX', 0),             # hit   – already cached
    ('LOAD', 'CX', 8),             # miss
    ('LOAD', 'CX', 4),             # hit
    ('LOAD', 'CX', 12),            # miss
    ('LOAD', 'CX', 0),             # hit

    # ALU on loaded value
    ('ADD',  'AX', 'CX'),

    # Branch prediction demo
    ('CMP',  'AX', 60),            # AX == 60? → DX = 0 (no, AX is 60+some loads)
    ('JEQ',  1),                   # predict taken  → likely wrong
    ('CMP',  'BX', 15),            # BX == 15? → DX = 1 (yes)
    ('JEQ',  1),                   # predict taken  → correct
    ('CMP',  'BX', 99),            # BX == 99? → DX = 0
    ('JEQ',  0),                   # predict not-taken → correct

    # Logical ops
    ('MOV',  'AX', 0b1010),
    ('MOV',  'BX', 0b1100),
    ('AND',  'AX', 'BX'),
    ('OR',   'AX', 'BX'),
    ('XOR',  'AX', 'BX'),

    # Unconditional jump (simulated)
    ('JMP',  None),
]


# ─────────────────────────────────────────────
# Run the simulation
# ─────────────────────────────────────────────

print("=" * 55)
print("  CPU Performance Analysis Tool  –  Simulation Run")
print("=" * 55)

for idx, instr in enumerate(instructions):
    print(f"  [{idx+1:02d}] Executing: {instr}")
    execute_instruction(instr)

print()
print("Registers after execution:", registers)
print(f"Total cycles consumed   : {execution_time}")
print(f"Cache hits              : {cache_hits}")
print(f"Cache misses            : {cache_misses}")
total_branches = branch_correct + branch_wrong
accuracy = (branch_correct / total_branches * 100) if total_branches else 0
print(f"Branch prediction acc.  : {accuracy:.1f}%  ({branch_correct}/{total_branches})")
print("=" * 55)


# ─────────────────────────────────────────────
# 4. VISUALIZE RESULTS
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("CPU Performance Analysis – Simulation Results", fontsize=14, fontweight='bold')

# ── Chart 1: Execution time per instruction (bar chart) ──
labels_  = [item[0] for item in cycle_log]
cycles_  = [item[1] for item in cycle_log]
colors_  = ['#e74c3c' if c >= 10 else '#3498db' for c in cycles_]

axes[0].bar(range(len(labels_)), cycles_, color=colors_, edgecolor='white')
axes[0].set_xticks(range(len(labels_)))
axes[0].set_xticklabels(labels_, rotation=45, ha='right', fontsize=7)
axes[0].set_ylabel("Cycles")
axes[0].set_title("Execution Time per Instruction\n(red = cache-miss penalty)")
axes[0].set_ylim(0, max(cycles_) + 2)

# add a simple legend
axes[0].bar(0, 0, color='#e74c3c', label='Cache miss (10 cycles)')
axes[0].bar(0, 0, color='#3498db', label='Normal (1–3 cycles)')
axes[0].legend(fontsize=7)

# ── Chart 2: Cache Hits vs Misses (pie chart) ──
pie_labels = ['Hits', 'Misses']
pie_values = [cache_hits, cache_misses]
pie_colors = ['#2ecc71', '#e74c3c']
axes[1].pie(pie_values, labels=pie_labels, colors=pie_colors,
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title(f"Cache Hit vs Miss Rate\n({cache_hits} hits, {cache_misses} misses)")

# ── Chart 3: Branch Prediction Accuracy (pie chart) ──
if total_branches > 0:
    bp_labels = ['Correct', 'Wrong']
    bp_values = [branch_correct, branch_wrong]
    bp_colors = ['#3498db', '#e67e22']
    axes[2].pie(bp_values, labels=bp_labels, colors=bp_colors,
                autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[2].set_title(f"Branch Prediction Accuracy\n({accuracy:.1f}%)")
else:
    axes[2].text(0.5, 0.5, "No branches\nexecuted",
                 ha='center', va='center', fontsize=12)
    axes[2].set_title("Branch Prediction Accuracy")
    axes[2].axis('off')

plt.tight_layout()
plt.savefig("cpu_performance_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved as: cpu_performance_results.png")