import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL = ["caltech3v", "caltech4v", "caltech5v", "caltech6v", "youtube", "coco", "voc2007"]
N_VIEWS = {"caltech3v": 3, "caltech4v": 4, "caltech5v": 5, "caltech6v": 6}


def ablation():
    arms = {"ddc_only": "--weights none", "uniform": "--weights uniform", "learned": "--weights learned",
            "fea_only": "--levels fea", "clu_only": "--levels clu"}
    return [(c, f"ablation/{a}", e) for c in ALL for a, e in arms.items()]


def descriptors():
    arms = {"drop_sep": "--desc_drop sep", "drop_ent": "--desc_drop ent", "drop_agree": "--desc_drop agree",
            "drop_mmd": "--desc_drop mmd", "qal_only": "--ire_terms qal", "cal_only": "--ire_terms cal",
            "qal_mean": "--qal_reduce mean"}
    return [(c, f"descriptors/{a}", e) for c in ALL for a, e in arms.items()]


def robustness():
    arms = {f"epsq{v}": f"--eps_q {v}" for v in [0.01, 0.05, 0.5, 1.0]}
    arms.update({"mmdbw0.5": "--mmd_bw_scale 0.5", "mmdbw2": "--mmd_bw_scale 2",
                 "wrc0.5": "--w_rc 0.5", "wrc2": "--w_rc 2", "wire0.5": "--w_ire 0.5", "wire2": "--w_ire 2"})
    return [(c, f"robustness/{a}", e) for c in ALL for a, e in arms.items()]


def sensitivity():
    jobs = []
    for c in ["caltech3v", "caltech4v", "caltech5v", "voc2007"]:
        jobs += [(c, f"sensitivity/alpha{v}", f"--alpha {v}") for v in [0.1, 0.3, 0.7, 0.9]]
        jobs += [(c, f"sensitivity/tau{v}", f"--tau_f {v}") for v in [0.05, 0.075, 0.15, 0.25, 0.4]]
    return jobs


def allocation():
    return [(c, f"allocation/{a}", f"--alloc {a}") for c in ["caltech4v", "caltech5v"]
            for a in ["uniform_mass", "permute"]]


def fixed(ws):
    return "--weights fixed --fixed_weights " + ",".join(map(str, ws))


def pairs():
    jobs = []
    for c in ["caltech4v", "caltech5v", "caltech6v"]:
        p = N_VIEWS[c] * (N_VIEWS[c] - 1) // 2
        jobs.append((c, "pairs/uniform", fixed([0.5] * p)))
        for i in range(p):
            ws = [0.0 if j == i else round(0.5 * p / (p - 1), 6) for j in range(p)]
            jobs.append((c, f"pairs/leave{i}", fixed(ws)))
            if c == "caltech4v":
                jobs.append((c, f"pairs/emph{i}", fixed([1.0 if j == i else 0.4 for j in range(p)])))
    return jobs


def degradation():
    conds = [("clean", "")]
    conds += [(f"gauss{v}", f"--corrupt_view 2 --corrupt_type gauss --corrupt_level {v}") for v in [0.25, 0.5, 1.0]]
    conds += [(f"shuffle{v}", f"--corrupt_view 2 --corrupt_type shuffle --corrupt_level {v}") for v in [0.25, 0.5, 1.0]]
    methods = {"bmcm": "", "uniform": "--weights uniform --fusion uniform"}
    return [(c, f"degradation/{m}_{k}", f"{me} {ce}".strip()) for c in ["caltech4v", "caltech5v"]
            for m, me in methods.items() for k, ce in conds]


def embedding():
    return [(c, "embedding", "--save_embedding_epochs 0,49,99,199") for c in ["voc2007", "youtube"]]


EXPERIMENTS = {"ablation": ablation, "descriptors": descriptors, "robustness": robustness,
               "sensitivity": sensitivity, "allocation": allocation, "pairs": pairs,
               "degradation": degradation, "embedding": embedding}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=list(EXPERIMENTS) + ["all"])
    p.add_argument("--datasets", default=",".join(ALL))
    p.add_argument("--runs_root", default="runs")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()
    names = list(EXPERIMENTS) if a.experiment == "all" else [a.experiment]
    keep = set(a.datasets.split(","))
    for name in names:
        for c, sub, extra in EXPERIMENTS[name]():
            if c not in keep:
                continue
            out = os.path.join(a.runs_root, sub, c)
            cmd = [sys.executable, "-m", "bmcm.train", "--config", f"configs/{c}.json", "--out", out] + extra.split()
            print(" ".join(cmd), flush=True)
            if a.dry_run:
                continue
            if not os.path.exists(os.path.join(ROOT, "data", f"{c}.mat")):
                print(f"skip: data/{c}.mat not found", flush=True)
                continue
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
