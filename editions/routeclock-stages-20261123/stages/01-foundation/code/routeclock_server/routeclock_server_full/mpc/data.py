"""生成独立物理场景，构建带有因果时间戳的训练和评估样本。"""
from pathlib import Path
import numpy as np
from routeclock.config import load_config
from routeclock.geometry import angles
from routeclock.predictors import Gate, Tracker, apply_gate
from routeclock.radio import Radio, array_gain
from routeclock.world import World

ROOT = Path(__file__).resolve().parents[1]
GRID = np.arange(0, 10.001, .25)

def publish(tracker, gate=None):
    """先发布只依赖历史的预测；离线标签生成和在线调度共用此入口。"""
    pkg = tracker.issue()
    if gate is not None:
        apply_gate(pkg, gate, pkg.features, "RC", "P")
    tracker.snapshots.clear()  # 本实验不调用 Chronos，其专用历史快照无需常驻内存。
    return pkg

def interpolate(rows, hs, grid=GRID):
    return np.column_stack([np.interp(grid, hs, col) for col in rows.T])

def inputs(pkg):
    """查询年龄、路线匹配、预测兑现误差、IMM 状态与链路几何均在发布时可得。"""
    columns = [1,2,3,4,5,6,7,8,9,12,13,14,16,17,18,19]
    rc = interpolate(pkg.centers["RC"], pkg.hs)
    return np.column_stack([GRID, interpolate(pkg.features[:,columns], pkg.hs),
                            np.linalg.norm(rc, axis=1), angles(rc)])

def labels(world, uid, pkg, radio):
    """仅在离线监督/事后审计调用。路线失效定义为纯路线指向首次越过参考窄束的 3 dB 界。"""
    ts = pkg.origin + GRID
    truth = np.stack([world.truth(uid, t)[0] for t in ts])
    shadow = np.array([world.shadow(uid, t) for t in ts])
    return link_labels(pkg,truth,shadow,radio)

def link_labels(pkg,truth,shadow,radio):
    """门控训练完成后用新融合中心重新构造速率标签。未来真值仅用于离线监督。"""
    route = interpolate(pkg.centers["P"], pkg.hs)
    reference = (32,32)
    gain = array_gain(angles(route), angles(truth), reference)
    failed = np.maximum.accumulate(10*np.log10(np.prod(reference)/gain) > 3.)
    # 实际通信指向用门控融合中心。每种候选束宽用自己的预测 SNR 选 MCS。
    center = interpolate(pkg.centers["RC"], pkg.hs)
    direction = angles(center)
    rates = []
    for pair in radio.pairs:
        mcs = np.array([radio.mcs(v) for v in radio.snr(center,direction,pair)])
        received = radio.snr(truth,direction,pair,shadow)
        rates.append(radio.goodput(received,mcs)[0]/1e6)
    return failed, np.column_stack(rates)

def make_scene(split, mission, cfg, progress=None):
    """每个 split 使用独立种子命名空间；预热期间错开观测，服务阶段的真值由仿真器保管。"""
    sim = load_config()
    sim["simulation"]["warmup_s"] = cfg["warmup_s"]
    sim["simulation"]["service_s"] = cfg["steps"]*cfg["dt"] if split=="test" else 0
    n = cfg["uavs"] if split in ("test","tube_cal") else cfg["training_uavs"]
    world = World(sim, f"server_full_v1/{split}", mission, n=n)
    # 每个独立校准网络只贡献一个事先选定的飞机和发布点，避免把重叠窗口冒充独立样本。
    rng=np.random.default_rng(73191+mission)
    uids=[int(rng.integers(n))] if split=="tube_cal" else range(n)
    trackers, states, samples = [], [], []
    for uid in uids:
        tr = Tracker(sim)
        times = np.arange(0., cfg["warmup_s"], 2.) + 2.*uid/n + sim["simulation"]["observation_cost_s"]
        for j,t in enumerate(times[times<cfg["warmup_s"]]):
            tr.observe(world.observe(uid,t))
            if t >= cfg["warmup_s"]-80:
                pkg = publish(tr)
                # 保留足够路线历史，每隔约 6 秒发布一个监督样本。相邻时域标签相关，
                # 因而训练、校准与测试按整个物理网络划分，而非打散这些行。
                if split!="test" and j%3==0:
                    ts=pkg.origin+GRID
                    truth=np.stack([world.truth(uid,s)[0] for s in ts])
                    shadow=np.array([world.shadow(uid,s) for s in ts])
                    samples.append((pkg,truth,shadow))
        trackers.append(tr)
        states.append(tr.evidence.archive[-1])
        if progress:
            progress(uid+1,n)
    if split=="test":
        return world,trackers,states
    return [samples[-1]] if split=="tube_cal" else samples
