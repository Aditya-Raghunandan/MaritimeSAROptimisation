"""
shutdown.py -- terminate a command-line run without interpreter shutdown.

WHY THIS EXISTS (measured 2026-09-17, both machines)

Anything that opens ARCO-ERA5 through gcsfs finishes its work, prints its
result, and then never exits -- on this laptop. Not slow: DEADLOCKED.

    laptop, Windows 11, CPython 3.13.1
        open the store            14.0 s
        fall off the end of main  killed at 180 s, 0.03 s of user CPU

    cluster, jaguar1, CPython 3.12.14 (micromamba env)
        open the store            15.2 s
        whole process             15.5 s wall, 3.6 s CPU, exit 0

So it is LOCAL ONLY, and it does NOT explain the eight-hour wall-clock kills
that job array 58632 took on 2026-09-16 -- the cluster does not have it. That
connection was tempting and is wrong; it is written down here so it does not
get made again.

WHAT IT IS NOT: every thread alive at exit is a daemon (`zarr_io`,
`asyncio_0`, `_poll_wrapper`), and daemon threads do not block shutdown. It is
an atexit handler -- `concurrent.futures.thread._python_exit` joins pooled
worker threads regardless of their daemon flag, and one of them never returns.

WHAT DOES NOT FIX IT: `ds.close()`. Measured, still killed at 90 s.

WHAT DOES: skipping interpreter shutdown entirely.

    open the store, then hard_exit(0)    14.1 s wall, exit 0

USE IT ONLY AS THE LAST STATEMENT OF A CLI `main()`, never in library code and
never before output is on disk. It runs no atexit handlers and no destructors,
so every file must already be written and closed. `sar.fetch.current` does not
need it -- that path is OPeNDAP/netCDF, not gcsfs, and exits normally.
"""

import os
import sys


def hard_exit(code: int = 0) -> None:
    """Flush stdout/stderr, then terminate immediately.

    Deliberately unconditional rather than `if sys.platform == "win32"`. A
    branch that only ever runs on one of the two machines is a branch that is
    only ever tested on one of them, and the cost on Linux at the end of a
    CLI -- where everything is already flushed and closed -- is nothing.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
