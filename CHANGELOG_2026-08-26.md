# 变更记录 — 2026-08-26（速度改良版）

> 目的：下次对话时读这个文件就能知道改了什么、为什么改。

## 背景

用 screen-use MCP 完成了一次真实的 Automatica 期刊投稿全流程（Papercept 注册 +
ORCID 注册 + 表单填写 + 上传 PDF 提交），30+ 步操作花了约 40 分钟。
事后做了完整的瓶颈分析（文件+行号级），本版本按优先级实施修复。

## 改了什么（9 个文件，97 测试全绿）

### 1. 截图不再撑爆上下文（最大元凶）
- `mcp_server.py`：`screenshot` 新增 `max_size`/`quality` 可选参数
- `perception/screen.py`：JPEG 超过 90KB 自动降质重编码（80→60→40）；
  缓存复用 `mss` 实例；缩放 LANCZOS→BILINEAR
- `config.py`：新增 `screenshot_max_bytes` 配置项（.env 可覆盖）
- 之前的问题：1280px q80 截图 base64 后 200-340KB，超过宿主 100KB 上限被截断，
  每次观察被迫绕路"存文件再读"，多花 1-2 轮 LLM 调用

### 2. 批量动作（消灭逐步往返）
- 新增 MCP 工具 `do_actions(actions, interval=0.3)`：一次调用执行一串动作，
  最后只返回一张截图
- `agent.py`：`run_task` 支持模型一次返回 `{"batch": [...]}` 多动作（向后兼容）；
  `max_steps` 默认 20→40

### 3. UIA 直读直写（绕开最慢路径）
- 新增 `get_element_text(element_id)` / `set_element_text(element_id, text)`
  （UIA ValuePattern），禁粘贴的输入框不再退化到逐键 press
- `uia_tree.py`：MAX_ELEMENTS 60→150 + 前台窗口优先配额；DFS 加 3000 节点访问预算；
  删掉了枚举时夹带的一次整屏截图（改 GetSystemMetrics）
- `list_ui_elements` 返回 `foreground_title` 字段

### 4. 逻辑修复
- `agent.py`：前台窗口判断不再用 elements[0]（按面积升序，是错的），
  改用 `foreground_title`
- `reflect.py`：屏幕指纹改用元素列表（剔除每轮重排的 id），
  不再用带 SoM 标注的截图（指纹会误变）
- 粘贴失败不再静默：`type_text` 粘贴后对 Edit 控件做 ValuePattern 比对，
  失败返回 ok=False 并建议换 `set_element_text`；剪贴板先备份后恢复

### 5. 执行器提速
- `pyautogui.PAUSE` 0.05→0.02；type_text 两处 sleep 0.1→0.05
- 纯 ASCII 文本走 `typewrite`（不再绕剪贴板），启用原死参数 `interval`
- 新增 `click_scaled(x, y)`：直接用截图上的坐标，内部乘 scale，宿主不用心算

### 6. VLM 调用
- OpenAI 客户端 `timeout=60`（防挂起卡死整个 MCP server）
- `max_tokens` 8192→1024（动作 JSON 够用，缩短长尾）

## 性能数据（实测冒烟）

- UIA 枚举：0.10s（修复前大页面可达数秒）
- 截图（含缩放+编码+自动降质）：0.08s
- 测试：97 passed（原 64 + 新增 33；2 个旧 executor 测试因行为按需求改变而更新）

## 注意

- MCP server 是长驻进程，**需要重启 Kimi CLI（或重连 MCP）才能用上新版工具**
- 论文已完成投稿（Papercept PIN 202064 / ORCID 0009-0008-0930-201X，
  详情见 D:\项目\Automatica\paper\SUBMISSION.md 和 D:\项目\Automatica\AGENTS.md）
