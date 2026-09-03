import numpy as np
import networkx as nx

from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Probe.WaveForm import WaveForm
from PySpice.Unit import u_V

from PySpice.Spice.NgSpice.Simulation import NgSpiceSubprocessCircuitSimulator

from .state_preserving_simulator import StatePreservingNgSpiceSimulator


class AbstractNetwork(Circuit):
    """
    Base class representing a learnable electrical network.

    Parameters
    ----------
    name : str
        Name of the SPICE circuit.
    con_graph : nx.Graph
        Connectivity graph whose edges correspond to learnable devices.
    node_cfg : tuple(np.ndarray, np.ndarray)
        (inputs, outputs) where each row is a pair of node indices [node_pos, node_neg].
        These are wired as behavioural voltage sources (B elements).
    epsilon : float
        Small regularisation value used by some subcircuits.
    backend : {"shared", "subprocess"}
        Which Ngspice backend to use.
    use_klu : bool
        If True, enable the KLU linear solver in ngspice via a simulator.options(...) call.
    """

    def __init__(
        self,
        name: str,
        con_graph: nx.Graph,
        node_cfg,
        epsilon: float = 1e-10,
        backend: str = "shared",
        use_klu: bool = True,
    ):
        self.name = name
        self.epsilon = float(epsilon)
        self.backend = str(backend)
        self.use_klu = bool(use_klu)

        # Initialise the underlying PySpice Circuit.
        super().__init__(name)

        # Node labels 0..N-1 to match the connectivity graph.
        self.__nodes__ = np.array([str(i) for i in range(con_graph.number_of_nodes())])

        # Behavioural voltage sources for inputs / outputs.
        # Each row of node_cfg[0] / node_cfg[1] is [node_pos, node_neg].
        self.inputs = [self.B(n + 1, *inds) for n, inds in enumerate(node_cfg[0])]
        self.outputs = [
            self.B(n + 1 + len(self.inputs), *inds) for n, inds in enumerate(node_cfg[1])
        ]

        # Index source used for DC sweeps over multiple examples.
        # We will always call dc(Vindex=slice(...)) in _solve().
        self.V("index", "index", 0, 1)

        # Choose and construct the simulator backend.
        if self.backend == "shared":
            # Single-process, shared-library backend with simple timing counters.
            self.cached_simulator = StatePreservingNgSpiceSimulator(self)
        elif self.backend == "subprocess":
            # Stock PySpice subprocess backend: spawns an ngspice process and parses its .raw output.
            # Do NOT pass pipe=... here; the class already sets pipe=True internally.
            self.cached_simulator = NgSpiceSubprocessCircuitSimulator(self)
        else:
            raise ValueError(f"Unknown backend '{self.backend}', expected 'shared' or 'subprocess'.")

        # Optional solver configuration: enable KLU and match alter-trainer
        # ngspice options where possible.
        try:
            opts = {
                "TEMP": 27,
                "TNOM": 27,
                "itl1": 40,
                "itl2": 40,
                "itl4": 6,
                "itl5": 60,
                "gmin": 1e-8,
                "reltol": 5e-3,
                "abstol": 1e-8,
                "vntol": 1e-5,
                "rshunt": 1e9,
            }
            if self.use_klu:
                opts["klu"] = True
            # All Ngspice-based simulators inherit CircuitSimulation.options().
            self.cached_simulator.options(**opts)
        except Exception:
            # Older ngspice / PySpice combinations may not support all options; fail soft.
            pass

    # ------------------------------------------------------------------
    # Low-level solve / predict API
    # ------------------------------------------------------------------
    def _solve(self, inputs, outputs=None):
        """
        Run a DC analysis for one or more input / (optionally) output samples.

        Parameters
        ----------
        inputs : array_like, shape (n_in,) or (n_samples, n_in)
            Input drive values for each behavioural source.
        outputs : array_like or None
            If provided, target output values (same shape as inputs for outputs).

        Returns
        -------
        analysis : PySpice.Probe.WaveForm.DcAnalysis
            DC operating points for all nodes for each sample.
        """
        inputs = np.transpose(inputs)
        if outputs is not None:
            outputs = np.transpose(outputs)

        assert len(inputs) == len(self.inputs)
        if outputs is not None:
            # Either outputs has the same shape as inputs, or it is (n_out, n_samples).
            assert outputs.shape == inputs.shape or outputs.shape[0] == len(self.outputs)

        # Normalise shapes: (n_in, n_samples)
        if inputs.ndim == 1:
            inputs = inputs.reshape((-1, 1))
            if outputs is not None:
                outputs = outputs.reshape((-1, 1))
        elif inputs.ndim != 2:
            raise ValueError(f"invalid input shape {inputs.shape}")

        n_examples = inputs.shape[1]

        # Build list of sources to drive: inputs (and outputs, if clamped).
        if outputs is not None:
            src_iter = list(zip(self.inputs, inputs)) + list(zip(self.outputs, outputs))
        else:
            src_iter = list(zip(self.inputs, inputs))

        # Program the behavioural sources as PWL functions of V(index).
        for source, v in src_iter:
            source.enabled = True
            if n_examples > 1:
                # Build "1 v1 2 v2 3 v3 ..." string.
                indexed_v = [str(val) for pair in zip(range(1, n_examples + 1), v) for val in pair]
                v_string = ", ".join(indexed_v)
                values_expr = f"{{pwl(V(index), {v_string})}}"
            else:
                # Single-sample: just set a constant value.
                values_expr = v[0]
            source.v = values_expr

        # Disable output drives when not clamping.
        if outputs is None:
            for source in self.outputs:
                source.enabled = False

        # One DC sweep over the index source sweeps all samples.
        analysis = self.cached_simulator.dc(Vindex=slice(1, n_examples, 1))

        # Provide an explicit "0" node as a WaveForm (PySpice doesn't always do this).
        analysis.nodes["0"] = WaveForm.from_unit_values("0", u_V(np.zeros(n_examples)))

        return analysis

    def solve(self, inputs, outputs=None):
        """Convenience wrapper returning a dense (n_nodes, n_samples) array."""
        analysis = self._solve(inputs, outputs)
        return np.array([u_V(analysis.nodes[str(i)]) for i in self.__nodes__])

    def predict(self, inputs):
        """
        Run the network in inference (free) mode and return output voltages.

        Parameters
        ----------
        inputs : array_like, shape (n_samples, n_in) or (n_in,)

        Returns
        -------
        preds : ndarray, shape (n_samples, n_out)
            Differential output voltages (V(pos) - V(neg)) for each
            configured output behavioural source.
        """
        analysis = self._solve(inputs)
        node0 = str(self.__nodes__[0])
        n_examples = len(analysis.nodes[node0])
        out = np.zeros((len(self.outputs), n_examples), dtype=float)
        for i, vsrc in enumerate(self.outputs):
            a, b = vsrc.node_names
            out[i] = u_V(analysis.nodes[a] - analysis.nodes[b])
        return out.T


class Ground_reference_edge(SubCircuit):
    """Single learnable edge with body tied to source by default."""

    __nodes__ = ("t_D", "t_S", "gnd")

    def __init__(self, name, circ, v_gs, r_shunt=1e16, epsilon=1e-6):
        super().__init__(name, *self.__nodes__)
        # Gate bias is implemented as a DC voltage source wrt ground.
        self.V(1, "t_G", "gnd", v_gs)
        # Very large shunt to avoid floating branches.
        self.R(1, "t_D", "t_S", r_shunt)
        # Ideal NMOS with body tied to source.
        self.MOSFET(1, "t_D", "t_G", "t_S", "t_S", model="Ideal")
        self.model("Ideal", "NMOS", level=2)

    def update(self, delta):
        # Adjust gate bias in-place; PySpice will see the new value on the
        # next simulation.
        self.V1.dc_value += float(delta)

    def get_val(self):
        return float(self.V1.dc_value)


class Ground_reference_edge_body(SubCircuit):
    """Edge with body explicitly tied to global ground."""

    __nodes__ = ("t_D", "t_S", "gnd")

    def __init__(self, name, circ, v_gs, r_shunt=1e16, epsilon=1e-6):
        super().__init__(name, *self.__nodes__)
        self.V(1, "t_G", "gnd", v_gs)
        self.R(1, "t_D", "t_S", r_shunt)
        # Bulk tied to global ground instead of source.
        self.MOSFET(1, "t_D", "t_G", "t_S", "gnd", model="Ideal")
        self.model("Ideal", "NMOS", level=2)

    def update(self, delta):
        self.V1.dc_value += float(delta)

    def get_val(self):
        return float(self.V1.dc_value)


class Ground_reference_edge_floating(SubCircuit):
    """Edge with a weakly-tied 'floating' bulk node."""

    __nodes__ = ("t_D", "t_S", "gnd")

    def __init__(self, name, circ, v_gs, r_shunt=1e16, epsilon=1e-6, r_bulk=1e12):
        super().__init__(name, *self.__nodes__)
        self.V(1, "t_G", "gnd", v_gs)
        self.R(1, "t_D", "t_S", r_shunt)
        # Floating bulk node t_B weakly tied to ground to avoid singularity.
        self.MOSFET(1, "t_D", "t_G", "t_S", "t_B", model="Ideal")
        self.R("bulk", "t_B", "gnd", r_bulk)
        self.model("Ideal", "NMOS", level=2)

    def update(self, delta):
        self.V1.dc_value += float(delta)

    def get_val(self):
        return float(self.V1.dc_value)


class GroundReferenceNetwork(AbstractNetwork):
    """
    Concrete network in which each graph edge is a single ideal NMOS device
    with a learnable gate voltage.
    """

    def __init__(
        self,
        name: str,
        con_graph: nx.DiGraph,
        node_cfg,
        body_to_ground: bool = False,
        floating_body: bool = False,
        epsilon: float = 1e-9,
        backend: str = "shared",
        use_klu: bool = True,
    ):
        super().__init__(
            name=name,
            con_graph=con_graph,
            node_cfg=node_cfg,
            epsilon=epsilon,
            backend=backend,
            use_klu=use_klu,
        )

        if floating_body:
            Edge = Ground_reference_edge_floating
        else:
            Edge = Ground_reference_edge_body if body_to_ground else Ground_reference_edge

        self.edges = []
        nodes_map = {n: i for i, n in enumerate(con_graph.nodes())}
        for n, (u, v, r) in enumerate(con_graph.edges(data="weight")):
            edge = Edge(f"e{n + 1}", self, r, epsilon=epsilon)
            self.subcircuit(edge)
            self.edges.append(edge)
            edge.circ = self.X(n + 1, f"e{n + 1}", nodes_map[u], nodes_map[v], 0)

    def update(self, updates):
        for edge, delta in zip(self.edges, updates):
            edge.update(delta)
