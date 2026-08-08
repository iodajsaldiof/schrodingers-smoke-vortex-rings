# Schrödinger's Smoke 双涡环实验

本仓库用于实现科研实践B题：利用不可压缩 Schrödinger 流（ISF）模拟两个同轴、同向涡环的交替穿越，并尝试从给定速度场中提取涡丝结构。

代码对应 Chern et al. 的基本流程：自由 Schrödinger 演化、逐点归一化、相位压力投影。当前仓库只包含数值代码与测试，不包含课程报告正文。

## 目录

- `src/isf_rings/`：核心数值模块。
- `scripts/run_leapfrogging.py`：双涡环演算入口；仅在手动执行时开始演算。
- `tests/`：不需要长时间模拟即可运行的小规模验证。
- `outputs/`：运行结果目录，已被 Git 忽略。

## 建议的首次运行方式

在仓库根目录安装依赖后，先使用开发网格进行短时检查：

```powershell
python -m pip install -e .
python scripts/run_leapfrogging.py --preset development --steps 48
```

题目给定的 `128 x 64 x 64` 网格和完整穿越周期应在核心测试通过后再运行。
