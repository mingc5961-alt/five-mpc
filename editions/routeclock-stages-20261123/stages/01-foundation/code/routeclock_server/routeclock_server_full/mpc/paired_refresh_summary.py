"""将逐事件结果与观测前后对账合并，生成各时域、各场景的描述统计。"""
import argparse
from pathlib import Path

import numpy as np

from .experiment import dump_csv
from .paired_refresh import read_csv


def summarize(folder):
    pairs = read_csv(folder/"paired_events.csv")
    audit = {(r["scene"], r["step"]): r for r in read_csv(folder/"refresh_value_audit.csv")}
    assert len(audit) == len(pairs)
    rows = []
    for r in pairs:
        extra = audit[r["scene"], r["step"]]
        assert r["uid"] == extra["uid"]
        rows.append({k: float(v) for k, v in (r | extra).items()})
    dump_csv(folder/"paired_with_audit.csv", rows)
    summaries, checks, patterns = [], [], []
    groups = {"all": rows, **{f"scene_{s}": [r for r in rows if r["scene"] == s] for s in (0, 1)},
              "at_slow_decision": [r for r in rows if r["offset"] == 0]}
    for label, group in groups.items():
        def col(field):
            return np.array([r[field] for r in group])
        for h in ("first", "matched", "locked", "1s", "3s", "5s"):
            d = col(f"delta_{h}_Mbit")
            summaries.append(dict(group=label, horizon=h, n=len(group), refresh_wins=int(sum(d>0)),
                wait_Mbit=col(f"wait_{h}_Mbit").mean(), refresh_Mbit=col(f"refresh_{h}_Mbit").mean(),
                mean_delta_Mbit=d.mean(), median_delta_Mbit=np.median(d),
                p25_Mbit=np.quantile(d,.25), p75_Mbit=np.quantile(d,.75),
                delta_narrow_ms=col(f"delta_{h}_narrow_ms").mean(),
                delta_observations=col(f"delta_{h}_observations").mean()))
        for pred, actual in (("pred_first_Mbit","delta_first_Mbit"),
                             ("pre_state_first_Mbit","delta_first_Mbit"),("post_obs_first_Mbit","delta_first_Mbit"),
                             ("pred_window_Mbit","delta_locked_Mbit"),("pre_state_window_Mbit","delta_locked_Mbit"),
                             ("post_obs_window_Mbit","delta_locked_Mbit"),("pred_window_Mbit","delta_matched_Mbit")):
            p, a = col(pred), col(actual)
            checks.append(dict(group=label,predictor=pred,actual=actual,predicted=p.mean(),realized=a.mean(),
                signed_bias_pred_minus_real=(p-a).mean(),mae=np.abs(p-a).mean(),
                correct_sign=int(sum((p>0)==(a>0))), predicted_loss_actual_win=int(sum((p<=0)&(a>0))),
                predicted_win_actual_loss=int(sum((p>0)&(a<=0))), predicted_wins=int(sum(p>0)), n=len(a)))
        one, three, five = (col(f"delta_{h}s_Mbit") for h in (1,3,5))
        patterns.append(dict(group=label, n=len(group), one_loss=int(sum(one<=0)),
            one_loss_three_win=int(sum((one<=0)&(three>0))), one_loss_five_win=int(sum((one<=0)&(five>0))),
            one_loss_three_five_win=int(sum((one<=0)&(three>0)&(five>0))),
            one_win_five_loss=int(sum((one>0)&(five<=0))),
            decision_pred_utility=col("pred_window_utility").mean(),
            updated_minus_assumed_target_Mbps=(col("observed_target_fresh_Mbps")-col("assumed_target_fresh_Mbps")).mean(),
            narrow_deadline_abs_error_s=np.abs(col("observed_narrow_deadline_s")-col("assumed_narrow_deadline_s")).mean(),
            extra_1to3_Mbit=(three-one).mean(), extra_3to5_Mbit=(five-three).mean()))
    for name, data in (("summary",summaries),("prediction_check",checks),("payback",patterns)):
        dump_csv(folder/f"{name}.csv", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/demo_paired_refresh"))
    summarize(parser.parse_args().out)
