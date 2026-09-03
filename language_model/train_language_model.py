#!/usr/bin/env python3
"""Full coupled-learning training + test of the reduced-vocab physics language model, on ngspice.

Every free-phase solve is a real ngspice DC operating point of the trained NMOS network, run across
a pool of worker processes for parallelism (PySpice holds the GIL, so processes -- not threads --
scale). The clamped phase pins each output branch toward its soft cross-entropy target with a stiff
clamp; the per-gate update is the local contrastive difference of the two phases. No backpropagation.

The corpus (four grammatical sentence shapes, incl. questions) is generated deterministically from
the grammar; a held-out split provides the test metrics. The defaults reproduce the shipped
`runs/clean` champion. Run `python train_language_model.py` from inside `language_model/`.
"""
from __future__ import annotations
import argparse, json, locale, os, shutil, sys, tempfile, time
from datetime import datetime
from multiprocessing import shared_memory, Value
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

os.environ["LM_FAMILIES"] = "stmt"
os.environ["LM_NO_OBJ_DET"] = "1"
_HERE = Path(__file__).resolve().parent          # language_model/
_REPO = _HERE.parent                             # repo root
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _find_libngspice():
    """Locate libngspice.so (the shared SPICE engine PySpice drives) portably."""
    import ctypes.util
    cands = [os.environ.get("NGSPICE_LIBRARY_PATH"), os.environ.get("LIBNGSPICE"),
             str(Path(sys.prefix) / "lib" / "libngspice.so"),
             "/usr/lib/libngspice.so", "/usr/local/lib/libngspice.so",
             "/opt/homebrew/lib/libngspice.dylib"]
    found = ctypes.util.find_library("ngspice")
    if found:
        cands.append(found)
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError("libngspice not found; set NGSPICE_LIBRARY_PATH to the shared library")


# each ngspice worker needs its own private handle to the shared library -> per-worker copies
_SRC = _find_libngspice()
_LIBDIR = Path(tempfile.gettempdir()) / "clln_lm_ngspice_libs"

# --- 1x mismatch additions: parametrized NMOSWRAP (per-instance kpval so beta is alterable via
# altermod) + per-device beta strength factors. VTO mismatch still uses the gate-shift trick; beta
# (kp) mismatch cannot be gate-shifted, so each device's kp is altered per branch. ---
KP_NOM_LM = 6.249608378152027e-05
NMOSWRAP_PARAM_KP = (
    ".subckt NMOSWRAP D G S B vth=0.75 kpval=6.249608378152027e-05\n"
    ".model m_loc nmos(level=1 vto={vth} gamma=1.09 phi=0.9499477708465783 tpg=1 "
    "kp={kpval} lambda=0.19998491299329302 rsh=73.21306042358299)\n"
    "M0 D G S B m_loc l=7.8e-6 w=0.138e-3 as=0.603e-8 ps=0.478e-3 ad=0.161e-8 nrd=.3 nrs=1\n"
    ".ends NMOSWRAP\n"
)


def _beta_factors_lm(n, sigma, seed, phase="free"):
    """Per-device kp strength multiplier 1+eps (dimensionless sigma); None if sigma<=0. Matches the
    common/ noise model construction (free and clamp use independent device draws)."""
    if sigma is None or sigma <= 0:
        return None
    salt = {"free": 0, "clamp": 1_000_003}.get(phase, 0)
    rng = np.random.default_rng(int(seed) + 2_000_006 + salt)
    return np.maximum(1.0 + rng.standard_normal(n) * float(sigma), 1e-6).astype(np.float64)


def _ensure_libs(n):
    _LIBDIR.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        dst = _LIBDIR / ("libngspice.so" if i == 0 else f"libngspice{i}.so")
        if not dst.exists() or dst.stat().st_size != _SRC.stat().st_size:
            shutil.copy2(_SRC, dst)


def branch_netlist(F):
    out, sink = F + 1, F + 2
    L = [".title b", NMOSWRAP_PARAM_KP.rstrip("\n"), ".options klu"]   # per-instance kp -> beta alterable
    for i in range(F):
        L.append(f"VIN{i} {i+1} 0 0")
    L += [f"RS1 {out} {sink} 1e9", f"VOUT0 {sink} 0 0"]
    for i in range(F):
        L += [f"VG{i} g{i} 0 2.0", f"RB{i} b{i} 0 10",
              f"X{i} {out} g{i} {i+1} b{i} NMOSWRAP vth=0.75 kpval={KP_NOM_LM:.12g}"]
    L += [".options gmin=1e-8 reltol=5e-3 abstol=1e-8 vntol=1e-5", ".options rshunt=1e9", ".op", ".end"]
    return "\n".join(L) + "\n"


# ===================== worker process state =====================
_W = {}


def _winit(counter, gates_name, x_name, vsink_name, dfree_name, dclamp_name,
           beta_free_name, beta_clamp_name, F, NB, MAXX, MAXV, nlist):
    with counter.get_lock():
        wid = counter.value
        counter.value += 1
    os.environ["NGSPICE_LIBRARY_PATH"] = str(_LIBDIR / "libngspice{}.so")
    import PySpice.Spice.NgSpice.Shared as ng_mod
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared
    def _load(self, verbose):
        locale.setlocale(locale.LC_NUMERIC, "C")
        if not getattr(ng_mod, "_c", False):
            ng_mod.ffi.cdef(open(Path(ng_mod.__file__).parent / "api.h").read()); ng_mod._c = True
        self._ngspice_shared = ng_mod.ffi.dlopen(self.library_path)
    NgSpiceShared._load_library = _load
    ng = NgSpiceShared(ngspice_id=wid % 60, send_data=False)
    ng.load_circuit(nlist)
    g = shared_memory.SharedMemory(name=gates_name)
    x = shared_memory.SharedMemory(name=x_name)
    v = shared_memory.SharedMemory(name=vsink_name)
    df = shared_memory.SharedMemory(name=dfree_name)
    dc = shared_memory.SharedMemory(name=dclamp_name)
    bf = shared_memory.SharedMemory(name=beta_free_name)
    bc = shared_memory.SharedMemory(name=beta_clamp_name)
    _W.update(ng=ng, F=F, out=F + 1,
              gates=np.ndarray((NB, F), np.float64, buffer=g.buf),
              X=np.ndarray((MAXX, F), np.float64, buffer=x.buf),
              vsink=np.ndarray((MAXV, NB), np.float64, buffer=v.buf),
              dfree=np.ndarray((NB, F), np.float64, buffer=df.buf),
              dclamp=np.ndarray((NB, F), np.float64, buffer=dc.buf),
              beta_free=np.ndarray((NB, F), np.float64, buffer=bf.buf),
              beta_clamp=np.ndarray((NB, F), np.float64, buffer=bc.buf),
              _shm=(g, x, v, df, dc, bf, bc))


def _chunk(ng, cmds):
    buf, n = [], 0
    for c in cmds:
        if buf and n + 2 + len(c) > 900:
            ng.exec_command("; ".join(buf)); buf, n = [], 0
        buf.append(c); n += (2 + len(c)) if n else len(c)
    if buf:
        ng.exec_command("; ".join(buf))


def _read_out(ng, out):
    s = ng.exec_command(f"print v({out})")
    # format: 'v(145) = 2.4e-01'
    try:
        return float(s.split("=")[-1])
    except Exception:
        for line in s.splitlines():
            if "=" in line and line.strip().startswith("v("):
                return float(line.split("=")[-1])
    return float("nan")


def _free_task(args):
    b, B = args
    ng, F, out = _W["ng"], _W["F"], _W["out"]
    gates, X, dfree, beta_free = _W["gates"], _W["X"], _W["dfree"], _W["beta_free"]
    # per-device VTO mismatch enters as a gate-voltage shift (vg - delta), exact for this device model.
    # free and clamp are independent physical device sets -> their own mismatch draws (dfree here).
    _chunk(ng, ["alter RS1=1e9", "alter VOUT0=0"] + [f"alter VG{i}={gates[b, i] - dfree[b, i]:.16f}" for i in range(F)])
    # 1x model: per-device beta (kp strength) can't be gate-shifted -> altermod each device's kp.
    if (beta_free[b] != 1.0).any():
        _chunk(ng, [f"altermod @m.x{i}.m0[kp]={KP_NOM_LM * beta_free[b, i]:.12g}" for i in range(F)])
    res = np.empty(B)
    for w in range(B):
        _chunk(ng, [f"alter VIN{i}={X[w, i]:.16f}" for i in range(F)])
        ng.run(); res[w] = _read_out(ng, out); ng.exec_command("destroy all")
    return b, res


def _clamp_task(args):
    b, B = args
    ng, F, out = _W["ng"], _W["F"], _W["out"]
    gates, X, vsink, dclamp, beta_clamp = _W["gates"], _W["X"], _W["vsink"], _W["dclamp"], _W["beta_clamp"]
    # clamp phase = the independent physical device set -> its own mismatch draw (dclamp).
    _chunk(ng, ["alter RS1=0.01"] + [f"alter VG{i}={gates[b, i] - dclamp[b, i]:.16f}" for i in range(F)])
    if (beta_clamp[b] != 1.0).any():   # 1x: independent clamp-device kp strength
        _chunk(ng, [f"altermod @m.x{i}.m0[kp]={KP_NOM_LM * beta_clamp[b, i]:.12g}" for i in range(F)])
    res = np.empty(B)
    for w in range(B):
        _chunk(ng, [f"alter VIN{i}={X[w, i]:.16f}" for i in range(F)] + [f"alter VOUT0={vsink[w, b]:.16f}"])
        ng.run(); res[w] = _read_out(ng, out); ng.exec_command("destroy all")
    return b, res


# ===================== solver (main side) =====================
class NgspiceSolver:
    def __init__(self, F, NB, n_workers, maxx=4096, maxv=64):
        _ensure_libs(min(60, n_workers))
        self.F, self.NB, self.maxx, self.maxv, self.n_workers = F, NB, maxx, maxv, n_workers
        self.g_shm = shared_memory.SharedMemory(create=True, size=NB * F * 8)
        self.x_shm = shared_memory.SharedMemory(create=True, size=maxx * F * 8)
        self.v_shm = shared_memory.SharedMemory(create=True, size=maxv * NB * 8)
        self.df_shm = shared_memory.SharedMemory(create=True, size=NB * F * 8)
        self.dc_shm = shared_memory.SharedMemory(create=True, size=NB * F * 8)
        self.bf_shm = shared_memory.SharedMemory(create=True, size=NB * F * 8)
        self.bc_shm = shared_memory.SharedMemory(create=True, size=NB * F * 8)
        self.gates = np.ndarray((NB, F), np.float64, buffer=self.g_shm.buf)
        self.X = np.ndarray((maxx, F), np.float64, buffer=self.x_shm.buf)
        self.vsink = np.ndarray((maxv, NB), np.float64, buffer=self.v_shm.buf)
        self.dfree = np.ndarray((NB, F), np.float64, buffer=self.df_shm.buf)
        self.dclamp = np.ndarray((NB, F), np.float64, buffer=self.dc_shm.buf)
        self.beta_free = np.ndarray((NB, F), np.float64, buffer=self.bf_shm.buf)
        self.beta_clamp = np.ndarray((NB, F), np.float64, buffer=self.bc_shm.buf)
        self.dfree[:] = 0.0; self.dclamp[:] = 0.0        # clean by default
        self.beta_free[:] = 1.0; self.beta_clamp[:] = 1.0  # nominal kp by default
        self._netlist = branch_netlist(F)
        self._build_pool()

    def set_mismatch(self, dfree, dclamp, beta_free=None, beta_clamp=None):
        self.dfree[:] = dfree; self.dclamp[:] = dclamp
        if beta_free is not None: self.beta_free[:] = beta_free
        if beta_clamp is not None: self.beta_clamp[:] = beta_clamp

    def _build_pool(self):
        counter = Value("i", 0)
        self.pool = ProcessPoolExecutor(
            max_workers=self.n_workers, initializer=_winit,
            initargs=(counter, self.g_shm.name, self.x_shm.name, self.v_shm.name,
                      self.df_shm.name, self.dc_shm.name, self.bf_shm.name, self.bc_shm.name,
                      self.F, self.NB, self.maxx, self.maxv, self._netlist))
        # warm the workers (also re-attaches them to the persistent shared memory)
        list(self.pool.map(_noop, range(self.n_workers)))

    def _rebuild_pool(self):
        try:
            self.pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self._build_pool()

    def _map_safe(self, fn, B):
        # shared-memory buffers (gates/X/vsink) persist across pool rebuilds, so a
        # crashed ngspice worker (BrokenProcessPool) is recoverable: rebuild + retry.
        from concurrent.futures.process import BrokenProcessPool
        tasks = [(b, B) for b in range(self.NB)]
        for attempt in range(5):
            try:
                out = np.empty((B, self.NB))
                for b, v in self.pool.map(fn, tasks, chunksize=4):
                    out[:, b] = v
                return out
            except (BrokenProcessPool, OSError) as e:
                print(f"[pool] {type(e).__name__}: {e}; rebuilding pool (attempt {attempt + 1}/5)", flush=True)
                self._rebuild_pool()
        raise RuntimeError("ngspice worker pool repeatedly broke (5 rebuilds failed)")

    def set_gates(self, gates):
        self.gates[:] = gates

    def free(self, X):
        B = X.shape[0]
        self.X[:B] = X
        return self._map_safe(_free_task, B)

    def clamp(self, X, vsink):
        B = X.shape[0]
        self.X[:B] = X
        self.vsink[:B] = vsink
        return self._map_safe(_clamp_task, B)

    def close(self):
        self.pool.shutdown()
        for s in (self.g_shm, self.x_shm, self.v_shm, self.df_shm, self.dc_shm, self.bf_shm, self.bc_shm):
            s.close(); s.unlink()


def _noop(i):
    return i


# ===================== training driver =====================
def main():
    import clm.helpers as T
    from clm import evaluate as EV
    from clm.embeddings import load_embedding_manifest, save_embedding_manifest
    from clm.grammar import (GRAMMAR_VERSIONS, CONTEXT_LEN, START_TOKENS, TERMINAL_PUNCT,
        SentenceRecord, as_train_test_bundle, build_windows_from_sentences, records_from_maybe_bundle,
        sample_training_sentences, split_sentence_pools, write_pool_bundle_files,
        write_train_test_pool_files)
    from clm.vocab import VOCAB

    BOS, OUT = T.BOS_ID, T.OUTPUT_DIM
    PHYS = np.array(T.PHYSICAL_OUTPUT_IDS)
    NB = len(PHYS)

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--embedding-json", default=str(_HERE / "embeddings" / "scibert_fa16.json"))
    ap.add_argument("--warmup-support-epochs", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--num-sentences", type=int, default=7000)
    ap.add_argument("--min-target-count", type=int, default=30)
    ap.add_argument("--max-train", type=int, default=200000)
    ap.add_argument("--max-val", type=int, default=1400)
    ap.add_argument("--max-test", type=int, default=1400)
    ap.add_argument("--train-test-only", action="store_true", default=True,
                    help="expose only train and the held-out split used for the test metrics (release default)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--onehot-gamma", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.035)
    ap.add_argument("--temp", type=float, default=0.0025)
    ap.add_argument("--workers", type=int, default=min(18, (os.cpu_count() or 4)))
    ap.add_argument("--benchmark-samples", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume-gates", default="")
    ap.add_argument("--start-epoch", type=int, default=1)
    ap.add_argument("--meas-sigma", type=float, default=0.0,
                    help="measurement read noise sigma in V (e.g. 2.25e-3), fresh per read")
    ap.add_argument("--vth-sigma", type=float, default=0.0,
                    help="per-device VTO mismatch sigma in V (e.g. 10e-3); free/clamp draw independently")
    ap.add_argument("--beta-sigma", type=float, default=0.0,
                    help="1x model: per-device beta/kp strength sigma (e.g. 0.005 = 0.5%%); 2E free/clamp independent")
    ap.add_argument("--vth-seed", type=int, default=0)
    ap.add_argument("--chip", "--vth-file", dest="vth_file", default="",
                    help="device-mismatch fingerprint npz (e.g. chips/chip_1.npz) to train under a chip")
    ap.add_argument("--meas-seed", type=int, default=-1,
                    help="read-noise stream seed; -1 = legacy (vth_seed+424242). Set explicitly for one canonical stream across chips")
    a = ap.parse_args()
    seed = a.seed
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    manifest = load_embedding_manifest(Path(a.embedding_json))
    M = T.manifest_matrix(manifest)
    dim = int(manifest.dim); F = CONTEXT_LEN * dim
    n_edges = NB * F

    run_dir = Path(os.environ["RUN_DIR"]) if "RUN_DIR" in os.environ else \
        _HERE / "train_output" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_embedding_manifest(run_dir / "embedding_manifest.json", manifest)

    bundle = split_sentence_pools(grammar_version=GRAMMAR_VERSIONS[0], seed=seed, dev_frac=0.15, test_frac=0.15)
    omitted_pool = None
    if a.train_test_only:
        # These grammar records were never used for either the published training
        # samples or the figure-evaluation windows. Retain them only transiently
        # to reproduce the original RNG stream; they are not written as a split.
        omitted_pool = [SentenceRecord.from_dict(item) for item in bundle["test_novel_pool"]]
        bundle = as_train_test_bundle(bundle)
        write_train_test_pool_files(run_dir, bundle)
    else:
        write_pool_bundle_files(run_dir, bundle)
    train_pool = records_from_maybe_bundle(bundle_path=run_dir / "sentence_pools.json", key="train_pool")
    metric_pool_key = "test_pool" if a.train_test_only else "dev_novel_pool"
    test_pool = records_from_maybe_bundle(bundle_path=run_dir / "sentence_pools.json", key=metric_pool_key)
    final_test_pool = None if a.train_test_only else records_from_maybe_bundle(
        bundle_path=run_dir / "sentence_pools.json", key="test_novel_pool")
    gv = train_pool[0].grammar_version

    sents, _ = sample_training_sentences(train_pool, num_sentences=a.num_sentences, curriculum_stage="core",
                                         min_target_count=a.min_target_count, seed=seed)
    tr_ctx, tr_y, _ = build_windows_from_sentences(sents)
    q_map = T.build_context_target_distributions(tr_ctx, tr_y)
    fb = T.unigram_distribution(tr_y)
    if a.max_train > 0:
        tr_ctx, tr_y = tr_ctx[:a.max_train], tr_y[:a.max_train]
    Qtr = np.stack([q_map.get(tuple(int(v) for v in c), fb) for c in tr_ctx]).astype(np.float64)
    tr_X = T.encode_contexts(tr_ctx, M, 0.0, 0.45).astype(np.float64)

    test_ctx, test_y, _ = build_windows_from_sentences([list(r.tokens) for r in test_pool])
    metric_limit = a.max_test if a.train_test_only else a.max_val
    if metric_limit and len(test_ctx) > metric_limit:
        s = rng.choice(len(test_ctx), metric_limit, replace=False); test_ctx, test_y = test_ctx[s], test_y[s]
    if omitted_pool is not None:
        omitted_ctx, _, _ = build_windows_from_sentences([list(r.tokens) for r in omitted_pool])
        if a.max_test and len(omitted_ctx) > a.max_test:
            rng.choice(len(omitted_ctx), a.max_test, replace=False)
    test_X = T.encode_contexts(test_ctx, M, 0.0, 0.45).astype(np.float64)

    # qmass/support metric = probability mass over ALL grammar-valid next-tokens (not just training-seen).
    # LM_GRAMMAR_SUPPORT points at a pickled {context_tuple: frozenset(valid_token_ids)} built from the
    # FULL (uncapped) grammar; fall back to the training-window support for any unseen context.
    _GS = None
    if os.environ.get("LM_GRAMMAR_SUPPORT"):
        import pickle as _pkl
        _GS = _pkl.load(open(os.environ["LM_GRAMMAR_SUPPORT"], "rb"))
        print(f"[qmass] grammar-support loaded ({len(_GS)} contexts) for the valid-next-token metric", flush=True)
    _gs_cache = {}
    def gsup(ctx_row):
        key = tuple(int(v) for v in ctx_row)
        m = _gs_cache.get(key)
        if m is None:
            vs = _GS.get(key) if _GS is not None else None
            if vs is None:
                m = q_map.get(key, fb) > 0
            else:
                m = np.zeros(OUT, bool)
                for t in vs: m[t] = True
            _gs_cache[key] = m
        return m

    if a.smoke:
        a.max_train = min(a.max_train, 240); tr_X, tr_y, Qtr = tr_X[:240], tr_y[:240], Qtr[:240]
        test_X, test_y = test_X[:400], test_y[:400]; a.warmup_support_epochs = 2; a.epochs = 2
        a.benchmark_samples = 120

    gates = np.random.default_rng(seed).uniform(2.5, 5.0, (NB, F))   # == shipped runs/clean vg_init
    vg_init0 = gates.copy()
    if a.resume_gates:
        gates = np.load(a.resume_gates).astype(np.float64)
        assert gates.shape == (NB, F), f"resume gates {gates.shape} != {(NB, F)}"
        print(f"[resume] loaded {a.resume_gates}; resuming at epoch {a.start_epoch}", flush=True)

    def softmax_full(V):
        z = V / a.temp; z[:, BOS] = -1e30; z -= z.max(1, keepdims=True)
        e = np.exp(z); return e / e.sum(1, keepdims=True)

    def assemble(Vbranch):
        full = np.zeros((Vbranch.shape[0], OUT)); full[:, PHYS] = Vbranch; return full

    print(f"=== ngspice coupled learning: vocab={OUT} branches={NB} edges={n_edges} F={F} "
          f"workers={a.workers} train_windows={tr_X.shape[0]} batch={a.batch} smoke={a.smoke} "
          f"meas_sigma={a.meas_sigma} vth_sigma={a.vth_sigma} vg_clip=[{T.VG_CLIP_LO},{T.VG_CLIP_HI}] ===", flush=True)
    (run_dir / "run_meta.json").write_text(json.dumps({
        "backend": "ngspice_coupled_learning_multiprocess", "edges": n_edges, "branches": NB, "F": F,
        "vocab": OUT, "embed_dim": dim, "temp": a.temp, "gamma": a.gamma, "onehot_gamma": a.onehot_gamma,
        "delta": a.delta, "batch": a.batch, "epochs": a.epochs, "warmup_support_epochs": a.warmup_support_epochs,
        "data_splits": ["train", "test"] if a.train_test_only else ["train", "dev", "test"],
        "metric_test_windows": int(len(test_y)),
        "workers": a.workers,
        "meas_sigma": a.meas_sigma, "vth_sigma": a.vth_sigma, "vth_seed": a.vth_seed,
        "vg_clip_lo": T.VG_CLIP_LO, "vg_clip_hi": T.VG_CLIP_HI, "argv": sys.argv}, indent=2))

    solver = NgspiceSolver(F, NB, a.workers)
    if a.vth_file:
        chip = np.load(a.vth_file)
        df = chip["vto_free"].reshape(NB, F).astype(np.float64)
        dc = chip["vto_clamp"].reshape(NB, F).astype(np.float64)
        bf = (chip["beta_free"].reshape(NB, F).astype(np.float64) if "beta_free" in chip.files
              else np.ones((NB, F)))            # 1x: per-device kp strength; legacy chips -> nominal
        bc = (chip["beta_clamp"].reshape(NB, F).astype(np.float64) if "beta_clamp" in chip.files
              else np.ones((NB, F)))
        solver.set_mismatch(df, dc, bf, bc)
        print(f"[noise] VTO+BETA mismatch LOADED from chip file {a.vth_file} "
              f"(beta std free|clamp {(bf.std())*100:.3f}|{(bc.std())*100:.3f}%) ", end="")
        print(f"[noise] VTO mismatch LOADED from chip file {a.vth_file} "
              f"(free|clamp std {df.std()*1e3:.2f}|{dc.std()*1e3:.2f} mV)", flush=True)
    rng_meas = np.random.default_rng(a.meas_seed if a.meas_seed >= 0 else a.vth_seed + 424242)

    def read_noise(V):
        if a.meas_sigma > 0:
            return V + rng_meas.normal(0.0, a.meas_sigma, V.shape)
        return V
    t_start = time.time()
    try:
        def eval_support(Xe, ye, ctxe):
            Vfull = assemble(read_noise(solver.free(Xe)))
            p = softmax_full(Vfull)
            Vm = Vfull.copy(); Vm[:, BOS] = -1e30; pred = Vm.argmax(1)
            sc = qm = ce = 0.0
            for i in range(len(ye)):
                sup = gsup(ctxe[i])                              # grammar-valid next-token support
                sc += int(sup[pred[i]]); qm += float(p[i][sup].sum())
                ce += -float(np.log(max(float(p[i][ye[i]]), 1e-12)))  # one-hot CE of the true next token
            return sc / len(ye), qm / len(ye), float((pred == ye).mean()), ce / len(ye)

        history = []
        if a.resume_gates and (run_dir / "history.json").exists():
            try:
                history = [h for h in json.loads((run_dir / "history.json").read_text())
                           if h.get("epoch", 0) < a.start_epoch]
            except Exception:
                history = []
        for epoch in range(a.start_epoch, a.epochs + 1):
            mode = "support" if epoch <= a.warmup_support_epochs else "onehot"
            g_ep = a.gamma if mode == "support" else a.onehot_gamma
            order = rng.permutation(tr_X.shape[0]); te = time.time()
            tr_ce = 0.0; tr_qm = 0.0; tr_n = 0   # running training-epoch avg CE + qmass (over presented windows)
            for s in range(0, len(order), a.batch):
                idx = order[s:s + a.batch]; Xb = tr_X[idx]; yb = tr_y[idx]
                solver.set_gates(gates)
                Vfree = read_noise(solver.free(Xb))           # (B,NB) measured (noisy) free voltages
                Vfull = assemble(Vfree); p = softmax_full(Vfull)
                if mode == "support":
                    q = Qtr[idx].copy()
                else:
                    q = np.zeros_like(Vfull); q[np.arange(len(yb)), yb] = 1.0
                q[:, BOS] = 0.0
                for j in range(len(yb)):                        # training-avg CE + qmass at the presented gates
                    supg = gsup(tr_ctx[idx[j]])                  # grammar-valid next-token support (matches val)
                    tr_qm += float(p[j][supg].sum()); tr_ce += -float(np.log(max(float(p[j][yb[j]]), 1e-12))); tr_n += 1
                Vclamp_full = Vfull + a.delta * (q - p)
                vsink = Vclamp_full[:, PHYS]
                # Stiff clamp: the clamped phase pins the output node to its soft-CE target (= vsink) while
                # the inputs are held, so every node entering the per-gate drop is known; the local
                # contrastive update follows directly. read_noise models the sense noise.
                Vclamp = read_noise(vsink)
                dfree = Vfree[:, :, None] - Xb[:, None, :]
                dclamp = Vclamp[:, :, None] - Xb[:, None, :]
                contrast = (dclamp ** 2 - dfree ** 2).mean(0)  # (NB,F)
                gates += -g_ep * contrast
                np.clip(gates, T.VG_CLIP_LO, T.VG_CLIP_HI, out=gates)
            rec = {"epoch": epoch, "mode": mode, "epoch_s": time.time() - te,
                   "train_ce": tr_ce / max(tr_n, 1), "train_qmass": tr_qm / max(tr_n, 1)}
            if epoch % a.eval_every == 0 or epoch == a.epochs:
                sa, qmv, ex, ce = eval_support(test_X, test_y, test_ctx)
                if a.train_test_only:
                    rec.update(test_support_acc=sa, test_qmass=qmv, test_exact_acc=ex, test_ce=ce)
                    print(f"[ep{epoch} {mode}] {rec['epoch_s']:.0f}s test support={sa:.3f} qmass={qmv:.3f} "
                          f"exact={ex:.3f} test_ce={ce:.4f} | train_ce={rec['train_ce']:.4f} "
                          f"train_qmass={rec['train_qmass']:.4f}", flush=True)
                else:
                    rec.update(support_acc=sa, qmass=qmv, exact_acc=ex, val_ce=ce)
                    print(f"[ep{epoch} {mode}] {rec['epoch_s']:.0f}s val support={sa:.3f} qmass={qmv:.3f} "
                          f"exact={ex:.3f} val_ce={ce:.4f} | train_ce={rec['train_ce']:.4f} "
                          f"train_qmass={rec['train_qmass']:.4f}", flush=True)
            else:
                print(f"[ep{epoch} {mode}] {rec['epoch_s']:.0f}s", flush=True)
            history.append(rec)
            np.save(run_dir / f"vg_epoch{epoch}.npy", gates)
            (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        np.save(run_dir / "vg_final.npy", gates)
        np.savez(run_dir / "gates.npz", vg_init=vg_init0, vg_final=gates)   # release-format endpoint gates
        try:
            _ep = [h["epoch"] for h in history]
            np.savez(run_dir / "curve.npz",
                     epoch=np.array(_ep),
                     train_ce=np.array([h.get("train_ce", np.nan) for h in history]),
                     test_ce=np.array([h.get("test_ce", np.nan) for h in history]),
                     train_qmass=np.array([h.get("train_qmass", np.nan) for h in history]),
                     test_qmass=np.array([h.get("test_qmass", np.nan) for h in history]),
                     test_support_acc=np.array([h.get("test_support_acc", np.nan) for h in history]),
                     test_exact=np.array([h.get("test_exact_acc", np.nan) for h in history]))
        except Exception as _e:
            print(f"[curve] skipped release curve.npz: {_e}", flush=True)
        solver.set_gates(gates)

        # ---- generation benchmark on ngspice ----
        def ngspice_generate(starts):
            N = len(starts)
            ctx = np.full((N, CONTEXT_LEN), BOS, dtype=int)
            out = [[] for _ in range(N)]; done = np.zeros(N, bool)
            grng = np.random.default_rng(seed + 999)
            for i, st in enumerate(starts):
                sid = T.TOKEN_TO_ID[st]; out[i].append(st); ctx[i] = np.append(ctx[i][1:], sid)
                if st in TERMINAL_PUNCT: done[i] = True
            for _ in range(11):
                act = np.flatnonzero(~done)
                if act.size == 0: break
                Xs = T.encode_contexts(ctx[act], M, 0.0, 0.45).astype(np.float64)
                Vfull = assemble(read_noise(solver.free(Xs))); Vfull[:, BOS] = -1e30
                pr = softmax_full(Vfull)
                for r, gi in enumerate(act):
                    y = int(grng.choice(OUT, p=pr[r] / pr[r].sum())); tok = VOCAB[y]
                    out[gi].append(tok); ctx[gi] = np.append(ctx[gi][1:], y)
                    if tok in TERMINAL_PUNCT: done[gi] = True
            return out

        references = (("test", test_pool),) if a.train_test_only else (("dev", test_pool), ("test", final_test_pool))
        for name, ref in references:
            sched = EV.forced_start_schedule(a.benchmark_samples, start_tokens=START_TOKENS)
            seqs = ngspice_generate(sched)
            rep = EV.evaluate_generated_sentences(seqs, grammar_version=gv,
                    train_pool_texts=EV._pool_text_set(train_pool),
                    reference_pool_texts=EV._pool_text_set(ref), reference_pool_name=name, start_tokens=sched)
            EV.save_report(run_dir / f"generation_validity_{name}_report_{a.benchmark_samples}.json", rep)
            print(f"NGSPICE {name.upper()} valid={rep['valid_rate']:.3f} heldout={rep.get('heldout_valid_rate'):.3f} "
                  f"completion={rep['completion_rate']:.3f} unique_valid={rep['unique_valid_count']} "
                  f"unique_heldout={rep.get('unique_heldout_valid_count')}", flush=True)
        print(f"\nTOTAL wallclock {(time.time()-t_start)/3600:.2f}h  run_dir={run_dir}", flush=True)
    finally:
        solver.close()


if __name__ == "__main__":
    main()
