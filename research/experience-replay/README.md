# research/experience-replay

经验回放（Experience Replay / Skill Library）调研与设计。2026-08-27。

| 文件 | 内容 |
|---|---|
| `DESIGN.md` | **主文档**：设计方案（宏 schema、录制/检索/回放三流水线、bailout 协议、实施路线、具身智能迁移路径） |
| `literature.md` | 文献调研：GUI agent 经验记忆、技能库、VLA 低效证据、分层架构、grounding（含具体数字与 URL） |
| `gaps.md` | 文献空白分析：本方案三件套（语义锚点+检查点+VLM 兜底）无先例论证、学术风险诚实清单 |

一句话：把 VLM 探索成功的轨迹编译成确定性宏，重复任务零模型回放，失败回退 VLM——
AppAgentX 已证明方向（SR 46.3→88.2%，每步 43.5s→17.5s），Windows/UIA 域是文献空白。
