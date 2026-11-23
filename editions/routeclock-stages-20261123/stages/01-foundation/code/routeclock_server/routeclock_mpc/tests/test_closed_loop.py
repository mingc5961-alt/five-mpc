"""利用已完成演示的独立测试起点，核对稀疏预算和可恢复输出的真实闭环入口。"""
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from mpc.data import ROOT
from mpc.experiment import simulate,report
from mpc.audit import verify

class ClosedLoopTests(unittest.TestCase):
    @unittest.skipUnless((ROOT/"runs/demo/test_0.pkl").exists() and
                         (ROOT/"runs/demo/risk.pkl").exists(),"先运行示例以生成独立测试场景和模型")
    def test_sparse_clock_and_resume(self):
        # 使用较低预算重新跑 20 个快轮；轮询和 MPC 共享最多四个名额。
        src=ROOT/"runs/demo"
        cfg=json.loads((src/"config.json").read_text(encoding="utf8"))
        cfg.update(steps=20,test_ids=[0],policies=["rotation","joint_mpc"],sensing_budget=.01)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for file in ["risk.pkl","test_0.pkl"]:
                (out/file).write_bytes((src/file).read_bytes())
            simulate(cfg,out)
            report(cfg,out)
            verify(cfg,out)
            folder=out/"scene_0"/"rotation"
            summary=json.loads((folder/"metrics.json").read_text(encoding="utf8"))
            self.assertEqual(summary["observations"],4)
            before=(folder/"ledger.csv").read_bytes()
            (folder/"in_progress.json").write_text("{}",encoding="utf8")
            simulate(cfg,out)
            self.assertEqual(before,(folder/"ledger.csv").read_bytes())
            self.assertFalse((folder/"in_progress.json").exists())
