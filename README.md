# stock-identity-pretrain

A Transformer-based Pre-train & Fine-tune framework that turns volatility noise into tradable signals.

两阶段金融时间序列框架：**先认股票身份，再预测涨跌**，并按不可妥协的实验协议做对照、消融、统计检验和经济评价。

核心假设：如果 Transformer Encoder 能从 30 日 K 线里认出「这是哪只美股」，它就必须学会每只股票的波动纹理。把这套特征冻住再接到次日涨跌头上，用 **MCC** 而不是容易被类别不平衡骗到的 Accuracy 来检验。

## 实验协议

| 项目 | 要求 |
|------|------|
| 种子 | 5 个 seed（0–4），所有指标报 mean ± std |
| 日志 | `runs/{exp_name}/{seed}.json`（config hash、指标、墙钟）+ `{seed}_preds.npz` |
| 入口 | `python experiments.py` → 自动写 `results.md` |
| 必须超过的对照 | 多数类、扁平特征 Logistic Regression（C 在验证集上搜）、1 个月动量符号规则、从零训练的 Transformer |
| 消融 | pretext：stock-ID / 掩码重建 / none；迁移：冻结 encoder / linear probe / 全量微调；特征：有无 `Volume_log`；是否做个股 z-score |
| 统计 | 按测试日 block-bootstrap（block=5，1000 次）给 MCC 95% CI；配对比较 ours 优于对照的 bootstrap 比例 |
| 表征 | 测试窗 embedding → UMAP + k-means，对 GICS 行业算 ARI；注意力热力图；图写入 `figures/` |
| 经济 | 每日多空：多头 P(up) 最高十分位、空头最低十分位、等权；年化 Sharpe、最大回撤、换手，以及 5bps / 10bps 单边成本后 Sharpe |

默认跑 **one-at-a-time** 消融（从 `ours` 每次只改一个因子）加上全部对照。全因子网格：`--matrix full`。

## 快速开始

```bash
git clone https://github.com/z26qin/stock-identity-pretrain.git
cd stock-identity-pretrain
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 接线冒烟：合成数据、1 seed × 2 epochs、整张 OAT 矩阵
python experiments.py --smoke

# 单测（防泄漏 scaler + 模型 shape + 出图）
python -m unittest discover -s tests -v

# 单次 demo（ours vs scratch）
python main.py --quick --synthetic
```

完整协议（65 只美股，2016–2024，5 seeds，预训练 100 / 微调 50 epoch）：

```bash
python experiments.py
```

数据优先 **AkShare**（`stock_us_daily`），失败回退 **yfinance**。原始 CSV 在 `data/raw/`，滑窗在 `data/processed/`。

## 代码结构

```
config.py          # T=30, 美股池, GICS, transfer_mode, 实验网格
dataset.py         # 滑窗 + 日期/次日收益/动量/行业；时间切分
model.py           # 三个 pretext path + transfer_mode
pretrain.py        # stock-ID 或 masked reconstruction
finetune.py        # frozen / linear probe / full
baseline.py        # Transformer from scratch
baselines.py       # majority / logreg / momentum
evaluate.py        # Accuracy, F1, MCC, AUC
stats.py           # block-bootstrap, 配对 win-rate
represent.py       # UMAP, ARI, attention heatmap
economics.py       # 十分位多空、成本后 Sharpe
analysis.py        # 表征 + 经济图 → figures/
experiments.py     # 全矩阵 → results.md
main.py            # 单次 ours vs scratch
utils.py           # 拉数、train-only scaler 统计、个股 z-score
tests/             # 防泄漏 + shape + 出图
```

## 数据与防泄漏

- 标的：美股（科技 / 金融 / 医疗 / 消费 / 能源工业等），GICS 行业用于 ARI
- 特征：`Open_pct`, `High_pct`, `Low_pct`, `Close_pct`, `Volume_log`
- 窗口 T=30；标签日 t 与 t+1 必须落在同一 split
- 标准化默认只在**训练集日频特征**上 fit 全局 scaler；消融可改为**按个股** train-only z-score
- **不按时间随机打乱**。DataLoader 只打乱训练窗

## 模型架构

- 2 层 Encoder，`d_model=64`，4 heads，GELU，last-token pooling
- `pretrain_task`：`stock_id` | `mask_recon` | `trend` / `trend_mlp`
- 掩码预训练：随机遮 15% 时间步，用 mask token，MSE 重建被遮特征
- 微调：冻结 encoder 只训涨跌头；linear probe 在 embedding 上用 Logistic Regression；full 解冻 encoder + MLP 头
- 梯度裁剪 `max_norm=1.0`

### 架构可视化 / Architecture Visualization

> **[▶ 在线交互式架构图](https://claude.ai/code/artifact/c295786c-8057-44cc-9813-a69804f98e53)**

一张交互式的单页可视化，展示 `StockTransformer` 的完整 forward pass 流程：

```
OHLCV Input ──▸ Linear Projection ──▸ + Sinusoidal PE ──▸ Encoder ×2 ──▸ Last-Token Pool ──▸ Task Head
  30 × 5            5 → 64              30 × 64           4H · GELU        30 → 1           → 65 / → 1
```

**功能亮点：**

| 功能 | 描述 |
|------|------|
| 动画数据流 | 金色光点沿 pipeline 流动，模拟 token 的前向传播过程 |
| 点击探索 | 点击任一组件，展开详细的内部结构、公式和参数说明 |
| 注意力扇形图 | Encoder 详情中绘制 SVG 弧线，展示 last-token 如何注意前序所有位置 |
| Pretrain ↔ Finetune 切换 | 切换任务头（Stock ID 65 分类 → Trend ↑↓ 二分类），冻结 encoder 时显示霜冻纹理 |
| 滚动 Ticker 带 | 65 只美股按 GICS 行业着色（Tech=蓝、Finance=金、Healthcare=绿…） |
| 真实参数 | 所有维度、特征名、参数量均来自 `model.py` 和 `config.py` |

## 怎么读指标

涨跌 Accuracy 经常卡在 ~50%。真正该看：

- **MCC ∈ [-1, 1]**：0 约等于随机。预训练如果把 MCC 从负值拉到正值，才说明学到了可迁移特征。
- **AUC > 0.5**：分数排序有信息量。
- **bootstrap CI 与配对 win-rate**：单次 MCC 的正负不够，要看对测试日重采样后是否稳定压过对照。
- **成本后 Sharpe**：没有换手约束的纸面 MCC 不等于可交易 pnl。

原项目示意数字（MCC −0.01 → +0.02）是故事锚点，**不保证在本仓库的数据/种子上复现**。以 `results.md` 为准。

## 电梯间说法

> 多数股价预测模型直接在涨跌标签上硬训，MCC 经常是负的：看起来有 Accuracy，其实比抛硬币还差。
> 身份预训练先让模型看 30 日 K 线，回答「这是哪只股票」。科技股和银行股的波动纹理不同，这个 N 分类会逼 Encoder 记微观结构，而不是记某一天的涨跌噪声。
> 然后冻住 Encoder 只训涨跌头。对照不只是从零训练的同一 Transformer，还包括多数类、扁平 Logistic、一个月动量规则。关键不是 Accuracy 多了零点几个点，而是 MCC 和成本后 Sharpe 有没有变成正的弱信号。
> 消融会检查这件事是不是只是成交量水平泄漏（去掉 `Volume_log`），表征上则看 embedding 能不能聚出 GICS 行业。
> 下一步可以把「股票 ID」换成「资产类别 ID」，用同一套框架做黄金、美债、美元指数、原油。
