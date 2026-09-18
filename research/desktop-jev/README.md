# research/desktop-jev

桌面版 Jev：训练小型校准决策模型，**加速**（而非替代）screen-use 的 VLM 循环。2026-09-18。

路线图源文件：`roadmap-kimi-2026-09-18.jpg`（Kimi 对话截图：从零训练类 Jev 模型用于 computer-use 的分阶段方案，5070Ti + 有限 API 预算约束版；owner 补充裁定：部分替代 VLM、加速为目的、不承担文字输入）。

| 文件 | 内容 |
|---|---|
| `TRAINING.md` | **训练设计**：数据 schema、观测序列化模板、四头架构与损失、显存/超参、评测消融、代码结构（DESIGN §9 的可执行级展开） |
| `DESIGN.md` | **主文档**：定位（加速器非替代者）、观测/动作空间（UIA 元素表 + 收窄动作集）、输出头架构、升级 VLM 协议、三级 cascade、数据三条腿、训练/评测、5070Ti 资源预算 |
| `literature.md` | 文献调研：Jev/RLCD、GUI agent 基线（UI-TARS/Aguvis/ShowUI）、grounding 小模型、环境/评测套（MiniWoB++/OSWorld/WindowsAgentArena）、校准方法（含具体数字与 URL） |
| `gaps.md` | 文献空白分析：桌面域校准决策模型无先例论证、与 grounding 工作的边界、学术风险诚实清单 |

一句话：把 VLM 视觉循环中占大头的"定位/导航"决策（点哪个、滚不滚、完成没）交给
0.5B–1.5B 的校准决策模型（Jev 架构思路），文本生成与低置信步骤升级 VLM——
Jev 已证明 System One 决策可以 70–500ms 完成（TypeSafe AI, 2026-09），
Windows/UIA 域的校准决策模型是文献空白。
