# Floating-Point Precision

## Overview

Floating-point precision determines how many significant digits your simulation carries through every calculation. BORES uses a global precision setting that affects all array allocations, PVT property computations, matrix assembly, and solver operations. The choice of precision involves a direct trade-off between computational speed (lower precision is faster) and numerical accuracy (higher precision reduces rounding errors).

In reservoir simulation, precision matters because the pressure and saturation equations involve quantities that span many orders of magnitude. Pressure values are in the thousands of psi, permeabilities range from millidarcies to darcies, and fluid compressibilities are on the order of $10^{-6}$ per psi. When these quantities are combined in transmissibility calculations, the intermediate values can lose significant digits if the precision is too low. However, for most engineering purposes, 32-bit precision provides more than adequate accuracy, and the 2x memory savings and faster computation make it the clear default choice.

BORES defaults to 32-bit (single) precision, which is the standard for field-scale reservoir simulation. You should only consider higher precision when you encounter specific numerical issues such as material balance errors, solver convergence problems in ill-conditioned systems, or when matching against analytical solutions that require tighter tolerances.

---

## Precision Levels

BORES provides three precision levels through simple function calls:

```python
import bores

bores.use_32bit_precision()  # numpy.float32 (default)
bores.use_64bit_precision()  # numpy.float64
bores.use_128bit_precision()  # numpy.float128
```

These functions set the global precision for all subsequent BORES operations. Call them before building your reservoir model, as the precision affects every array that BORES creates from that point forward.

### 32-bit Precision (Default)

```python
bores.use_32bit_precision()
```

32-bit floating-point numbers (IEEE 754 single precision) carry approximately 7 significant decimal digits with a range from $\pm 1.2 \times 10^{-38}$ to $\pm 3.4 \times 10^{38}$. This is the default and recommended setting for nearly all reservoir simulation work. It uses half the memory of 64-bit precision, which means faster array operations, better cache utilization, and the ability to run larger models in the same amount of RAM.

For a typical 100,000-cell model, 32-bit precision uses roughly 400 KB per property grid versus 800 KB at 64-bit. The difference multiplies across the dozens of property grids maintained during simulation, making 32-bit models noticeably faster overall.

### 64-bit Precision

```python
bores.use_64bit_precision()
```

64-bit floating-point numbers (IEEE 754 double precision) carry approximately 15 significant decimal digits with a range from $\pm 2.2 \times 10^{-308}$ to $\pm 1.8 \times 10^{308}$. This precision level is appropriate when you need higher accuracy or when the 32-bit solver shows convergence difficulties.

Consider 64-bit precision when:

- You observe material balance errors growing over many time steps
- The iterative solver takes significantly more iterations than expected
- You are matching against an analytical solution or a commercial simulator that uses double precision
- Your model has extreme property contrasts (permeability ratios above $10^6$, pressure differences above $10^4$ psi)
- You are performing sensitivity studies where small numerical differences matter

The performance penalty is typically 30 to 80% compared to 32-bit, depending on the problem size and the proportion of time spent in array operations versus solver iterations.

### 128-bit Precision

```python
bores.use_128bit_precision()
```

128-bit extended precision carries approximately 18 to 33 significant digits depending on the platform (on most x86 systems, `numpy.float128` is actually 80-bit extended precision, not true IEEE 754 quad precision). This level is rarely needed in practice and is primarily useful for debugging numerical issues or for reference solutions.

!!! warning "Platform Support"

    Not all platforms support `numpy.float128`. On Windows, this type may not be available. On Linux and macOS with x86 processors, it typically maps to 80-bit extended precision (not true 128-bit). Check `numpy.finfo(numpy.float128)` to see the actual precision on your system. Additionally, most Numba-compiled functions do not support 128-bit types, so some performance-critical code paths may fall back to Python, resulting in significantly slower execution.

---

## Temporary Precision with Context Manager

The `with_precision()` context manager lets you temporarily change precision for a specific block of code without affecting the global setting. This is useful when you want to run most of your simulation at 32-bit but need higher precision for a specific operation.

```python
from bores import with_precision
import numpy as np

bores.use_32bit_precision()  # Global default

# Build model at 32-bit precision
model = bores.reservoir_model(...)

# Run a specific analysis at 64-bit
with with_precision(np.float64):
    # Everything inside this block uses float64
    high_precision_result = some_analysis(model)

# Back to float32 outside the block
```

The context manager is thread-safe because it uses Python's `ContextVar` mechanism. Each thread maintains its own precision setting, so changing precision in one thread does not affect other threads. This is important if you are running multiple simulations in parallel.

---

## Querying the Current Precision

You can check and inspect the current precision setting at any time:

```python
from bores.precision import get_dtype, get_floating_point_info

# Get the current dtype
current = get_dtype()
print(current)  # <class 'numpy.float32'>

# Get detailed floating-point information
info = get_floating_point_info()
print(f"Precision: {info.precision} decimal digits")
print(f"Range: {info.min} to {info.max}")
print(f"Smallest difference: {info.eps}")
```

The `get_floating_point_info()` function returns a `numpy.finfo` object with details about the current precision level, including the number of significant digits, the representable range, and the machine epsilon (the smallest number such that $1.0 + \epsilon \neq 1.0$).

---

## Choosing the Right Precision

For the vast majority of reservoir simulation work, 32-bit precision is the right choice. Here is a decision guide:

| Situation | Recommendation |
|---|---|
| Standard field-scale simulation | 32-bit (default) |
| Matching commercial simulator results | 64-bit |
| Extreme permeability contrasts (> $10^6$) | 64-bit |
| Material balance errors accumulating | Try 64-bit |
| Solver taking many iterations (> 100) | Try 64-bit |
| Debugging or reference solutions | 128-bit (if available) |
| Large models where memory matters | 32-bit |
| Maximum performance | 32-bit |

!!! tip "Start with 32-bit"

    Always start with 32-bit precision. If you encounter numerical issues, switch to 64-bit and see if the problem resolves. If it does, the issue was precision-related. If it does not, the root cause is elsewhere (grid quality, time step size, solver configuration) and higher precision will not help.

---

## Impact on Performance

The table below shows approximate performance differences between precision levels for typical reservoir simulation operations:

| Operation | 32-bit | 64-bit | 128-bit |
|---|---|---|---|
| Memory per property grid (100K cells) | 400 KB | 800 KB | 1.6 MB |
| Array arithmetic | Baseline | 1.3 to 1.5x slower | 5 to 10x slower |
| Sparse matrix solve | Baseline | 1.5 to 2.0x slower | Not supported by most solvers |
| Total simulation time | Baseline | 1.3 to 1.8x slower | Much slower (if it works) |

These numbers are approximate and depend on hardware, problem size, and the ratio of compute-bound to memory-bound operations. On modern CPUs with AVX2 or AVX-512 SIMD instructions, 32-bit operations can process twice as many values per instruction as 64-bit, making the difference particularly pronounced for array-heavy operations.
