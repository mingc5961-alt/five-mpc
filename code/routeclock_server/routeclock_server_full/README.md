# RouteClock 独立训练与联合调度平台

上传完整源码包，解压，在目录内建 Python 3.12 环境即可。从空目录开始生成航线、训练门控、训练并校准融合角误差管、训练路线失效概率与条件通信速率模型，再做多机双时间尺度 MPC 闭环。包内没有旧模型权重和旧实验数据。

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py all --config configs/smoke.json --out runs/smoke --workers 4
~~~

正式研究请先阅读 [服务器使用手册](服务器使用手册.md) 和 [模型架构与主要函数](模型架构与主要函数.md)。正式配置为 30 个测试网络，每个 16 机。所有预测器及控制器支持 CPU；GPU 不是启动条件。
