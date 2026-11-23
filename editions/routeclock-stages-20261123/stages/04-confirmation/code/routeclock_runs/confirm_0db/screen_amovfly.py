from pathlib import Path
import numpy as np
import pandas as pd

root = Path("work/amovfly")
metadata = pd.read_csv(root / "Flight_info.csv", encoding="utf-8-sig")
fields = ["time", "gps_x", "gps_y", "gps_z", "real_lat", "real_long"]
rows = []
for _, flight in metadata.iterrows():
    group = str(flight["Data Dir"])
    flight_id = str(flight["FlightName"])
    file = root / group / group / f"{flight_id}.csv"
    result = dict(flight_id=flight_id, group=group, session=flight["Date time"], route=flight["Route"], source=str(file), status="missing")
    rows.append(result)
    if not file.is_file():
        continue
    values = pd.read_csv(file, usecols=fields)[fields].to_numpy(dtype=float)
    time, z, lat, lon = values[:, 0], values[:, 3], values[:, 4], values[:, 5]
    ground = np.nanmedian(z[time < time[0] + 1])
    valid = np.isfinite(values).all(axis=1) & (z >= ground + 5) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180) & (lat != 0) & (lon != 0)
    delta = np.diff(time)
    breaks = np.flatnonzero(~(valid[:-1] & valid[1:] & (delta > 0) & (delta <= .25))) + 1
    edges = np.r_[0, breaks, len(time)]
    result["status"] = "no_continuous_window"
    for lo, hi in zip(edges[:-1], edges[1:]):
        if not valid[lo] or time[hi-1] - time[lo] < 370.02:
            continue
        start = time[lo]
        end = min(hi-1, np.searchsorted(time[lo:hi], start+370.02)+lo)
        result.update(status="quality_candidate", first_t=start, warm_end=start+180, score_end=start+360, truth_end=start+370.02, gap_max=float(np.max(delta[lo:end])))
        break
table = pd.DataFrame(rows)
table.to_csv("work/amovfly_quality_candidates.csv", index=False)
print(table.groupby(["group", "status"]).size().to_string())
print("Quality candidates require periodic-route and site verification before becoming experiment windows.")
