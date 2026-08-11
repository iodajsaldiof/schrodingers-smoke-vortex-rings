# MATLAB 独立版本

本目录保存 B 题任务 2 的 MATLAB 基准实现，与 `src/isf_rings/` 下的 Python
实现相互独立。

## `reference/`

该子目录保存 Chern et al. 在 Schrödinger's Smoke 项目页公开的 MATLAB 示例
原文件：`ISF.m`、`TorusDEC.m`、`Particles.m`、`example_leapfrog.m`。它们保持
原样，仅作为论文算法 1--3 的基准；请在提交或公开仓库前核查作者的发布许可
与引用要求。

## 可运行文件

### `run_reference.m`

这是作者 `example_leapfrog.m` 的可诊断复现：保留作者的共心、异半径双环和背景
速度，用于核对论文示例。它会保存关键帧、环心与半径、局部环量、能量和离散散度：

```matlab
result = run_reference('outputs/作者示例_测试', 2000)
```

### `run_leapfrogging.m`

这是 B 题任务 2 的**主展示算例**，复现 Chern et al. 论文 5 的 Figure 4 / 作者
`example_leapfrog.m`：共心、异半径 `R=[1.5,0.9]`、同向同轴的双涡环。论文报告
该设置在 2000 步后仍约有四个交替穿越周期。与作者原示例不同，本项目会保存可信
环心轨迹和定量周期判据：

```matlab
result = run_leapfrogging
```

### `run_equal_rings.m`

这是题目建议的参数工况：两条半径 `R=1`、轴向间距 `d=0.6`、同向同轴的涡环。
它应作为参数敏感性对照；若输出 `order_cycle_detected=0`，则该参数组合不能作为
“完成交替穿越”的主结果。当前的提取器会把涡核合并帧标为 `track_resolved=0`，
而不会把两个数值峰误当成两条可辨识的涡环。

```matlab
result = run_equal_rings
% 或指定一个全新的输出目录和最长步数
result = run_equal_rings('outputs/等半径双涡环_测试', 2400)
```

每次正式运行应使用**全新的输出目录**。这样旧的 `state_*.mat` 不会混入新的图和
CSV；程序检测到旧结果时会报错，绝不会自动删除任何文件。

未手动指定目录时，程序会自动在 `outputs/` 下创建中文目录，例如
`双涡环完整交替穿越_20260811_153000`、`等半径双涡环_...` 和
`双涡环_网格时间步收敛性_...`。末尾时间戳用于区分不同次运行。

## 输出与判据

输出目录位于仓库的 `outputs/`，已由 `.gitignore` 排除。主要文件如下：

- `state_*.mat`：关键时刻波函数和两组材料标记坐标；两次顺序交换也会自动保存。
- `ring_tracks.csv`：欧拉相位涡量峰的盒内/连续展开坐标、半径、量子绕数和可信度。
- `material_ring_tracks.csv`：两组分别从初始涡环播撒的拉格朗日材料标记的环心与半径；
  这是判定交替穿越的主数据，能跨越几何峰值合并维持环身份。
- `cycle_summary.csv`：以材料标记为主的候选交换、可信交换与顺序恢复时间，
  并同时记录欧拉峰值的交换计数作交叉核对。
- `solver_diagnostics.csv`：离散散度、Dirichlet 总能量、速度动能和密度约束误差。
- `diagnostics.png`：环心轨迹、半径、不可压残差、能量/环量的合图。
- `visuals/`：多个关键时刻的三维离散相位涡量等值面、青/洋红两组材料标记叠加图，
  以及 `(x,r)` 剖面图。

`track_rings.m` 使用与论文 Algorithm 3 一致的边相位一形式和离散外微分，
先将涡量压缩为 `(x,r)` 图，再通过峰值匹配描述当前几何。当峰间距离不足网格
可分辨尺度时，`track_resolved=0`。主周期判据借鉴论文 5 的粒子推进，利用两组
初始涡丝材料标记保持身份；`order_cycle_detected=1` 仅在两次交换前后都有稳定
材料轨迹时成立，不以单张图或峰值跳变替代。

### `run_convergence.m`

该函数会分别运行粗/基准/细网格和时间步减半算例，比较顺序恢复时间、末态能量与
散度残差。它计算量很大，建议在正式 B 题算例已成功后单独执行：

```matlab
summary = run_convergence
% 如需研究题目建议 R=1,d=0.6 工况：
summary = run_convergence('', 85, 'equal_rings')
```

已有 MAT 关键帧时，可直接在 MATLAB 控制台运行：

```matlab
visualize_states
```

函数会优先搜索名称含“完整交替穿越”的主算例；主算例不存在时，才退回到仓库
`outputs/` 下最新且包含 `state_*.mat` 的实验目录。若需指定某次计算，使用
`visualize_states(result.output_path)`。
