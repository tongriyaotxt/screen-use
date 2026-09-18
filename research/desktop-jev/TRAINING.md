# desktop-jev 训练设计（TRAINING）

> 2026-09-18 · 配套阅读：`DESIGN.md`（总体架构）、`literature.md`（方法出处）
> 状态：设计提案，未实现。本文是 §9「训练与评测」的可执行级展开，供评审后进入实现。

## 0. 一页总览

```
数据（三条腿）                    模型（单次前向四头）              产出
┌─ 腿1 真实轨迹 20% ──┐
├─ 腿2 mock 合成 70% ─┼→ Qwen3-0.6B/1.7B + 特殊token ─→ 操作8类 / 元素151槽 / span / 置信度
└─ 腿3 教师难例 10% ──┘   序列化观测 ≤4K token          ↓
                                                    temperature scaling → τ 升级阈值
                                                    ↓
                                          离线：acc / ECE / 延迟
                                          在线：升级率 / VLM调用数 / 成功率
```

**训练目标不是"会操作电脑"，而是"在快路径上校准地操作电脑"**——
准确率决定快路径有没有用，校准（ECE）决定升级协议可不可靠。两者分开验收。

## 1. 数据 schema

每步一条样本（JSONL），与轨迹采集（阶段 0）落盘格式一致，训练时直接消费：

```json
{
  "sample_id": "tracehash:step_idx",
  "goal": "在计算器算 78*9 并把结果写到记事本",
  "foreground_title": "计算器",
  "history": [{"action": "CLICK", "element_name": "Seven", "ok": true}],
  "elements": [
    {"id": 1, "control_type": "ButtonControl", "name": "One",
     "bbox": [412, 330, 446, 352], "window_title": "计算器"}
  ],
  "label": {
    "action": "CLICK",
    "element_id": 34,
    "text_span": null,
    "source": "vlm_trace | mock_oracle | teacher_local | teacher_api",
    "quality": "gold | silver"
  }
}
```

要点：

- `label.source` 全程追踪，**按来源分层报告指标**（合成准确率再高也不算数，真实子集才算）。
- 教师标注的为 silver，人工/程序化 oracle 的为 gold；评测集**只用 gold**。
- 负样本（BLOCKED/DONE）来自：轨迹末尾的 DONE、VLM 循环中 `classify_stuck` 触发点、
  合成环境显式构造（元素不存在的查询）。这三类天然少，**过采样到 ≥10%**。
- TYPE 样本额外存 `text_span = [goal_char_start, goal_char_end]`；
  span 标注由启发式生成（输入文本在 goal 中的字面/归一化匹配），匹配不上则该样本
  降级为 BLOCKED 标签——**宁可教模型认输，不教它编文本**。

## 2. 观测序列化（模型输入模板）

与 DESIGN §4 一致，但为指针头加入**锚点特殊 token**（这是实现层的关键决定）：

```
<|goal|>在计算器算 78*9 并把结果写到记事本<|end_goal|>
<|window|>计算器
<|hist|>[CLICK Seven ✓]<|end_hist|>
<|elem|>#1 ButtonControl "One" 412 330 446 352
<|elem|>#2 ButtonControl "Two" ...
<|decide|>
```

- 新增 6 个特殊 token（`<|goal|> <|end_goal|> <|window|> <|hist|> <|end_hist|> <|elem|> <|decide|>`），
  embedding 行追加 + 随机初始化，参与训练。Qwen3 tokenizer 有预留空位，成本极低。
- **`<|elem|>` 的隐状态 = 该元素的向量**（元素指针头的打分对象）；
  **`<|decide|>` 的隐状态 = 全局决策向量**（操作头/置信度头的输入）；
  **goal 区间 token 的隐状态 = span 头的打分对象**。
  三个头的"指针"全部落在显式锚点上，不依赖脆弱的"按位置数 token"。
- 长度上限 4096 token，超长截断策略：先砍 history 滑窗（保最近 4 步），
  再砍元素表（保前 120 条，配额逻辑与 `list_ui_elements` 一致），**永不截 goal**。

## 3. 模型与四个头

```
backbone: Qwen3-0.6B（v0）/ Qwen3-1.7B（v1），bf16，单次前向，无解码
  h_decide   = H[<|decide|>]            # 全局决策向量
  h_elem_i   = H[<|elem|>_i], i=1..150  # 元素向量
  h_goal_t   = H[goal 区间 token]        # span 向量

操作头:   op_logits    = W_op · h_decide                    # 8 类
元素头:   elem_logits_i = w_e · h_elem_i  (+ no_elem 槽 = w_e · h_decide)   # 151 槽
槽位头:   start_t = w_s · h_goal_t, end_t = w_o · h_goal_t   # goal token span
置信度头: conf = σ(w_c · h_decide + b)                       # 标量
```

设计理由：

- **单次前向、无解码循环**：Jev 范式的核心，<100ms 的前提。输出空间全部预先枚举
  （8 操作 × 151 元素槽），schema 符合性数学保证——不可能输出元素 #152。
- 元素头做 **pointer softmax 而非坐标回归**：候选集内选择天然类型安全，
  且与运行时 `list_ui_elements` 的输出一一对应（id 重排不变，因为训练/推理
  用的是同一段枚举代码——**train/serve 一致性靠复用 `uia_tree.py` 保证**）。
- 置信度头与操作头联训但**标签独立**：y_conf = 1{操作+元素同时预测对}，
  用 Brier 损失（见 §4）——这是 RLCD"校准优先"精神的 SFT 近似。
- 头参数总量 <20M，不到 backbone 的 4%；砍掉 lm_head 还省显存。

### 显存预算（5070Ti 16GB，bf16 + gradient checkpointing + 8bit AdamW）

| 项 | 0.6B | 1.7B |
|---|---|---|
| 权重 | 1.2 GB | 3.4 GB |
| 梯度 | 1.2 GB | 3.4 GB |
| 优化器（8bit） | 1.2 GB | 3.4 GB |
| 激活（ckpt, bs=8, 4K） | ~4 GB | ~5 GB |
| **合计** | **~8 GB ✅** | **~15 GB ⚠️ 贴边** |

1.7B 贴边 → 先 0.6B 跑通管线立基线，1.7B 作为准确率不够时的升级项（可开 ZeRO-offload 兜底）。

## 4. 损失函数

```
L = L_op + 1.0·L_elem + 0.5·L_span + 0.5·L_conf

L_op   = CE(op_logits, action)                          # 全部样本
L_elem = CE(elem_logits, element_id)                    # 有目标元素的样本
L_span = CE(start) + CE(end)                            # 仅 TYPE 样本
L_conf = Brier(conf, 1{op 和 elem 均预测对})             # 全部样本
```

- L_conf 的标签**由模型自身预测与 gold 的比较生成**（在线自举），
  这正是"校准"与"准确"的解耦点：模型可以对同一状态同时输出
  "我认为是 CLICK #34"和"我只有 40% 把握"。
- 类别不均衡：DONE/BLOCKED/TYPE 过采样；CLICK 类内部按 control_type 分层
  （防止模型只学"按钮好点"）。
- 权重 0.5/0.5 是起点，消融项（§7）。

## 5. 数据配比与课程

| 阶段 | 数据 | 目的 |
|---|---|---|
| SFT-A（合成打底） | 腿 2 mock 10K+ 步（70%）+ 腿 1 真实（30% 混入防过拟合合成风格） | 学会动作语义与 UIA 结构先验 |
| SFT-B（真实对齐） | 腿 1 全部 + 腿 3 教师难例，真实占比 ≥70% | 对齐真实分布；mock 只留 30% 正则 |
| 校准（不训练） | 留出集 500 步 gold | temperature scaling + 等张回归 + τ 选择 |

配比 70/20/10 是起点假设，**第一条消融曲线就是合成:真实比**（1:0 / 7:3 / 3:7 / 0:1）。

## 6. 训练超参（起点）

| 项 | 值 | 备注 |
|---|---|---|
| precision | bf16 | |
| optimizer | AdamW 8bit，lr 1e-4，cosine，warmup 3%，weight_decay 0.01 | 头与 backbone 同 lr（头 <4% 参数，不分组） |
| batch | 8 × grad_accum 4 = 32 | 4K token 上限 |
| epochs | SFT-A 3，SFT-B 2 | 每 epoch 在留出集测 acc + ECE，早停 |
| 预估时长 | 10K 步 × 0.6B：5070Ti 约 1–2h/epoch | 全程单机可完成，总实验周期以天计 |

## 7. 评测与消融

**离线（每次训练完自动跑）**：

| 指标 | 定义 | 及格线（起点，可修） |
|---|---|---|
| op acc | 操作分类准确率（真实 gold 子集） | ≥85%（低于此快路径无意义） |
| joint acc | op+element 同时对 | ≥80% |
| ECE / Brier | 校准误差（temperature scaling 后） | ECE ≤0.05 |
| 延迟 | 5070Ti，4K ctx，p50/p95 | p50 <100ms |
| selective 曲线 | 覆盖率-风险曲线（扫描 τ） | **核心产出**：τ 选覆盖率 70–75% 对应点 |

**在线（接入升级协议后，对照 = 现状纯 VLM 循环）**：
升级率 / 每任务 VLM 调用数 / 端到端成功率 / 壁钟时间，四指标同测。
**升级率超过 35% 或成功率下降 >5 个点 = 阶段 4 不通过。**

**消融（按优先级）**：
1. 合成:真实配比（§5）——决定腿 2 还要不要扩。
2. 去置信度头（τ 退化为固定升级率）——验证校准值不值 0.5 的 loss 权重。
3. 去 UIA（bbox 置零）/ 去 history——观测各字段的贡献。
4. 0.6B vs 1.7B——决定 backbone 终点。

## 8. 代码结构（实现时的目录，本次不写）

```
training/                     # 独立目录，不动 screen_use 包
├── dataset.py                # JSONL → 序列化 → tokenize（复用 screen_use.perception 的枚举代码保证 train/serve 一致）
├── model.py                  # Qwen3 + 四头（<150 行）
├── train.py                  # SFT-A/B（HF Trainer 或 200 行自写 loop）
├── calibrate.py              # temperature scaling + 等张 + τ 选择 + selective 曲线
├── eval_offline.py           # §7 左表
└── serve.py                  # 推理封装：观测进 → (action, elem, conf) 出，供 VisualLoop 调用
```

依赖新增：`torch transformers scikit-learn`（训练侧独立 venv 或 extra，
`screen-use` 运行时包不背 torch——serve 阶段再议，ONNX/llama.cpp 量化导出是备选）。

## 9. 与阶段路线的对应

| DESIGN §10 阶段 | 本文对应 |
|---|---|
| 0 轨迹采集 | §1 schema 即落盘格式，采集改造以本文为准 |
| 1 数据基线 | §5 真实占比统计 + kill criterion（op acc 上限预估） |
| 2 mock 合成 | §5 腿 2 配比，tkinter 窗口的 oracle 标签即 §1 label |
| 3 SFT v0 | §3–§6 全部 |
| 4 校准+接入 | §7 selective 曲线 → τ；serve.py 接 VisualLoop |
| 5 教师蒸馏 | §1 silver 样本回流 SFT-B，迭代 |

## 10. 诚实边界

- **8 类/151 槽的容量上限**：动作集扩到 HOTKEY/PRESS 或元素表 >150 时，
  当前头设计需要重新验证（类别数翻倍 ≈ 数据需求翻倍）。
- **置信度头的在线自举标签有循环依赖**：模型越准，y_conf 越偏 1。
  缓解：每轮 SFT-B 用最新 checkpoint 重算 conf 标签（自蒸馏式迭代，最多 2 轮）。
- **span 头是最可能砍掉的组件**：若 TYPE 样本 <5% 或 span 标注质量差，
  v0 直接砍 span 头、TYPE 全升级——损失覆盖率但大幅降低复杂度，
  此决定在阶段 1 数据统计出来后做。
- 所有超参均为起点而非结论；本文件随实验结果修订，修订记录追加在文件尾。
