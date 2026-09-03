"""State-preserving NgSpice simulator (revised)

This version is a thin wrapper around PySpice's NgSpiceSharedCircuitSimulator.
It keeps a single ngspice shared-library context for the lifetime of the
simulator and exposes simple cumulative timing counters around dc() calls.

IMPORTANT: we do NOT override or re-pass the 'pipe' argument here. The parent
NgSpiceSharedCircuitSimulator already ensures that the shared-library backend
is used by calling NgSpiceCircuitSimulator.__init__(..., pipe=False, ...).
"""

import time
import logging

from PySpice.Spice.NgSpice.Simulation import NgSpiceSharedCircuitSimulator

_module_logger = logging.getLogger(__name__)


class StatePreservingNgSpiceSimulator(NgSpiceSharedCircuitSimulator):
    _logger = _module_logger.getChild("StatePreservingNgSpiceSimulator")

    def __init__(self, circuit, **kwargs):
        # DO NOT touch 'pipe' here. NgSpiceSharedCircuitSimulator.__init__
        # already calls super().__init__(circuit, pipe=False, **kwargs).
        super().__init__(circuit, **kwargs)

        # Cumulative timing counters (seconds). Only t_total is actively
        # updated; the others are kept for compatibility with older logging.
        self.t_total = 0.0
        self.t_super = 0.0
        self.t_remove_destroy = 0.0
        self.t_load = 0.0
        self.t_reset = 0.0
        self.t_run = 0.0
        self.t_plot = 0.0
        self.t_dc_analysis = 0.0
        self.n_runs = 0

    def dc(self, *args, **kwargs):
        """Run a DC analysis and accumulate total wall-clock time."""
        t0 = time.time()
        analysis = super().dc(*args, **kwargs)
        dt = time.time() - t0
        self.t_total += float(dt)
        self.n_runs += 1
        return analysis
