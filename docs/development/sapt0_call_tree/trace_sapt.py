"""Trace the cqed_scf call tree of calc.sapt0_components() for one backend.

Mirrors run_worker() in water_methylamine_benchmark_resources.py.
Records Python calls in cqed_scf, plus the first-level calls out of cqed_scf
into psi4 (Python or C) so the boundary is visible.
"""
import json, os, sys, threading
import numpy as np
import psi4
from cqed_scf import CQEDCalculator, CQEDConfig

backend, basis, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, "/Users/jfoley19/Code/sapt_demonstration_data/Result_3_Computational_Performance")
from water_methylamine_benchmark_resources import DIMER

PKG = os.path.dirname(__import__("cqed_scf").__file__)
psi4.set_num_threads(1); psi4.set_memory("4 GB")
psi4.core.set_output_file(os.path.join(os.path.dirname(out), f"psi4_{backend}.out"), False)
dense = backend == "full_eri"
config = CQEDConfig(lambda_vector=np.array([0.0, 0.0, 0.05]), omega=0.1,
    psi4_options={"basis": basis, "scf_type": "pk" if dense else "df",
                  "e_convergence": 1e-10, "d_convergence": 1e-8},
    reference="rhf", functional=None, density_fitting=not dense, charge=0,
    multiplicity=1, dispersion_policy="none", debug=False, quiet=True)
calc = CQEDCalculator(config=config)
mol = psi4.geometry(DIMER)

def node(name): return {"name": name, "count": 0, "children": {}}
root = node("ROOT"); stack = [root]; ext_depth = [0]

def label(frame):
    co = frame.f_code
    fn = os.path.relpath(co.co_filename, os.path.dirname(PKG))
    qual = getattr(co, "co_qualname", co.co_name)
    return f"{qual}  [{fn}:{co.co_firstlineno}]"

def push(name):
    parent = stack[-1]
    ch = parent["children"].get(name)
    if ch is None: ch = parent["children"][name] = node(name)
    ch["count"] += 1; stack.append(ch)

def prof(frame, event, arg):
    infile = frame.f_code.co_filename.startswith(PKG)
    if ext_depth[0]:            # inside an external (psi4) call: only track nesting
        if event == "call": ext_depth[0] += 1
        elif event == "return":
            ext_depth[0] -= 1
            if ext_depth[0] == 0: stack.pop()
        return
    if event == "call":
        if infile:
            if frame.f_code.co_name in ("<genexpr>", "<listcomp>", "<dictcomp>", "<lambda>"): 
                push(label(frame)); return
            push(label(frame))
        else:
            caller = frame.f_back
            if caller and caller.f_code.co_filename.startswith(PKG) and "psi4" in frame.f_code.co_filename:
                mod = frame.f_globals.get("__name__", "?")
                push(f"{mod}.{getattr(frame.f_code,'co_qualname',frame.f_code.co_name)}  (psi4, python)")
                ext_depth[0] = 1
    elif event == "return":
        if infile: stack.pop()
    elif event == "c_call" and infile:
        slf = getattr(arg, "__self__", None)
        smod = type(slf).__module__ if slf is not None else ""
        mod = getattr(arg, "__module__", None) or ""
        if "psi4" in mod or "psi4" in smod:
            if slf is not None and smod.startswith("psi4"):
                nm = f"psi4.core.{type(slf).__name__}.{arg.__name__}"
            else:
                nm = f"psi4.core.{arg.__name__}"
            p = stack[-1]; nm += "  (psi4, C++)"
            ch = p["children"].get(nm) or p["children"].setdefault(nm, node(nm))
            ch["count"] += 1

kw = dict(integral_backend=backend, include_cavity_terms=True, monomer_reference_frame="monomer_com")
sys.setprofile(prof)
comps = calc.sapt0_components(mol, **kw)
sys.setprofile(None)

def conv(n): return {"name": n["name"], "count": n["count"], "children": [conv(c) for c in n["children"].values()]}
json.dump({"backend": backend, "basis": basis, "tree": conv(root),
           "components": {k: float(getattr(comps, k)) for k in ("elst10","exch10","ind20","exch_ind20","disp20","exch_disp20","total")}},
          open(out, "w"), indent=1)
print("done", backend)
