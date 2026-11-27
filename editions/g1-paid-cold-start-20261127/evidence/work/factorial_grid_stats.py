from factorial_stats import B, D, best, json, np, write

def compare():
    pts={a:[] for a in ("G1","R1")}
    for k in range(8):
        fingerprint=None
        for arm in pts:
            z=[]
            for m in (.4,.85):
                name=f"sensitivity_grid_{k}_G1_{m:.2f}.json" if arm=="G1" else f"run_confirm_{k}_R1_{m:.2f}.json"
                d=json.loads((D/name).read_text());r=d["result"]
                assert fingerprint is None or d["world"]==fingerprint
                fingerprint=d["world"]
                assert abs(sum(r["ledger_s"])-180)<1e-6 and abs(len(d["events"])*.0031-r["ledger_s"][0])<1e-7
                assert r["probes_per_user"]==9000 and abs(sum(r["bits"])/180e6-r["G_mbps"])<1e-6
                z.append([r["G_mbps"],r["R"]*100])
            pts[arm].append(z)
    pts={a:np.asarray(z) for a,z in pts.items()}
    matched={a:best(z.mean(0),7.) for a,z in pts.items()}
    boot={a:best(z[B].mean(1),7.) for a,z in pts.items()}
    ok=np.isfinite(boot["R1"])&np.isfinite(boot["G1"])
    assert all(np.isfinite(v) for v in matched.values()) and ok.any()
    delta=matched["R1"]-matched["G1"]
    lo,hi=np.quantile(boot["R1"][ok]-boot["G1"][ok],[.025,.975])
    write("samegrid_result.csv",["risk_pct","R1_Mbps","G1_Mbps","delta","lo","hi","feasible"],[[7.,matched["R1"],matched["G1"],delta,lo,hi,ok.mean()]])

if __name__=="__main__":compare()
