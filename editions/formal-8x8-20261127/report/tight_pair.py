"""Post-hoc two-model tight-budget comparison, reusing the archived policy and bootstrap."""
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "run"))
from inference_stats import boot
from tables import workpoint


def main():
    observed = pd.read_csv(HERE.parent / "run/results/all_main/endpoints.csv")
    test = observed.query("w >= 100 and b == 'T' and a in ('Cnative', 'G8P')")
    mean = test.groupby(["a", "m"])[["risk", "mbps", "sense", "narrow"]].mean()
    intervals = pd.DataFrame(boot(test, "T"))
    rows = []
    for policy in ("confirmation_common", "both_tuned"):
        target, chosen = workpoint(test, "Cnative")
        if policy == "confirmation_common" and abs(mean.loc[("Cnative", 1.), "risk"] - target) < 1e-9:
            chosen["Cnative"] = (1., 1., 0.)
        for method, (left, right, prob) in chosen.items():
            result = (1 - prob) * mean.loc[(method, left)] + prob * mean.loc[(method, right)]
            ci = intervals.query("policy == @policy and a == @method and metric == 'mbps'").iloc[0]
            rows.append(dict(policy=policy, method=method, target_risk_pct=target, left=left, right=right,
                             prob_right=prob, **result.to_dict(), delta_mbps_lo95=ci.lo95,
                             delta_mbps_hi95=ci.hi95, infeasible_fraction=ci.infeasible_fraction))
    pd.DataFrame(rows).to_csv(HERE / "tight_pair.csv", index=False)


if __name__ == "__main__":
    main()
