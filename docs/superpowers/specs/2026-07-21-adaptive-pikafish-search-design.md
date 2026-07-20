# Pikafish 自适应低延迟搜索设计

**日期：** 2026-07-21  
**分支：** `dev-1.0`

## 目标

在不修改 Pikafish C++ 搜索核心、不降低现有输入校验和错误处理的前提下，缩短默认分析等待时间，同时保留明确的搜索硬上限和较高的局面分析强度。

默认配置下，程序应让 Pikafish 使用已有的自适应时间管理，根据最佳着稳定度、评分变化和节点集中度提前结束稳定局面的搜索；复杂局面仍可使用更多时间，但不得超过用户设置的硬上限。

## 已确认的根因

当前适配层只发送 `go movetime N`。Pikafish 在该模式下按硬截止时间停止，不进入基于 `wtime`/`btime` 的自适应时间管理，因此默认 10 秒搜索通常完整等待约 10 秒。

适配层还把线程数设置为 `os.cpu_count()`。本机 Python 返回 12，但 Pikafish 只检测到 8 个可用处理器。官方 `speedtest` 中，8 线程约为 2.93M NPS，12 线程约为 2.94M NPS，吞吐几乎不变，标称 5 秒搜索总时长却从 5.12 秒增加到 6.07 秒。

界面的“停止分析”目前只设置 Python 事件并丢弃最终结果，没有向正在搜索的 UCI 引擎发送 `stop`，所以后台线程仍会等到搜索预算耗尽。

## 方案选择

采用 Pikafish 原生自适应时间管理，不在 Python 中实现自定义 PV 稳定启发式。

搜索命令同时提供：

- `wtime` 和 `btime`：均设为硬上限的两倍，为 Pikafish 提供自适应分配所需的虚拟剩余时间。
- `movestogo 10`：让引擎按十个未来着次分配虚拟时间。
- `movetime`：保持用户设置的绝对硬上限。

例如用户设置 10 秒时发送：

```text
go wtime 20000 btime 20000 movestogo 10 movetime 10000
```

本机四线程实测中，标准开局和已报告中盘均约 4.2 秒返回；中盘推荐着与固定 10 秒搜索一致。该测量只作为性能证据，不作为正式 Elo 证明。

## 组件与数据流

### `pikafish_engine.py`

`PikafishEngine._configure_strength()` 将线程数限制为：

```python
min(8, max(1, os.cpu_count() or 1))
```

Hash 保持 128 MB，`MultiPV` 保持 1。当前四秒量级搜索不需要增加 Hash；单主变化线是最低延迟配置。

`PikafishEngine.get_best_move(position, movetime)` 继续执行：

```text
ucinewgame
isready
position ...
go ...
等待 bestmove
```

保留 `ucinewgame`。每次截图分析都是缺少完整历史的独立局面，清空置换表和搜索历史优先保证重复局面及规则判定的一致性；实测也没有发现保留旧表能稳定提升相邻局面的一秒搜索深度。

新增一个最小停止方法，向仍在运行的进程写入 `stop` 并刷新 stdin。它不等待 `bestmove`，由当前搜索线程继续读取并正常收尾。

### `app.py`

用户设置继续表示毫秒级硬上限，输入范围仍为 1～60 秒。UI 标签从“搜索秒”改为“最长秒”，默认值仍为 10。

`AssistantApp.stop()` 在设置 `stop_event` 后，读取当前引擎引用并调用引擎停止方法。搜索结束后的结果仍因 `stop_event` 被丢弃，保持现有取消语义。

### `README.md`

文档说明：

- “最长秒”是绝对上限，稳定局面可能提前返回。
- 默认 10 秒上限在当前测试机器上通常约 2～5 秒返回，但实际时间依硬件和局面变化。
- 程序最多使用 8 个逻辑线程。
- 实际 Hash 为 128 MB，而不是旧文档中的 512 MB。

## 并发和错误处理

搜索仍由现有后台工作线程执行，stdout 仍由单独读取线程写入队列。停止命令只负责通知 UCI 引擎；`bestmove`、超时、进程退出诊断和引擎重启逻辑保持现状。

停止可能与引擎启动短暂竞争。`AssistantApp.best_move()` 已在创建或复用引擎前检查 `stop_event`；新增停止方法还必须检查进程存在且未退出。停止失败不得覆盖原有关闭和进程回收逻辑。

不新增第三方依赖，不增加配置文件，不修改嵌套 `pikafish` 仓库。

## 测试设计

先写失败测试，再实现最小代码：

1. `_configure_strength()` 在 `os.cpu_count()` 为 12 时发送 `Threads value 8`，低于 8 时仍使用实际数量。
2. `get_best_move(..., movetime=5000)` 发送同时包含虚拟时钟、自适应步数和硬上限的 `go` 命令。
3. 引擎停止方法只在进程存活时写入并刷新 `stop`。
4. `AssistantApp.stop()` 设置事件并通知当前引擎。
5. 默认搜索上限仍为 10 秒，输入范围仍为 1～60 秒。

完成后运行：

```text
python3 -m unittest -v test_assistant.py
python3 -m py_compile app.py capture.py board_recognition.py pikafish_engine.py game_state.py
```

并用真实 Pikafish 对标准开局和已报告中盘各执行一次短搜索，确认返回合法 UCI 着法、硬截止时间有效且停止命令能结束搜索。

## 验收标准

- 默认 10 秒设置不再机械等待完整 10 秒；稳定局面可由 Pikafish 自适应提前结束。
- `movetime` 继续提供不可超过的硬上限。
- 最多使用 8 个线程，避免当前机器上的过度订阅。
- 点击停止会向正在搜索的引擎发送 `stop`，而不只是隐藏最终结果。
- 不修改 Pikafish 搜索、评估、剪枝或 NNUE 源码。
- 现有测试全部通过，新增行为有回归测试覆盖。

## 明确不做

- 不重写 Alpha-Beta、NNUE、剪枝或多线程搜索。
- 不增加开局库、残局库、GPU 模型或 MCTS。
- 不保留跨独立截图搜索的置换表。
- 不新增线程数、Hash 或自适应系数 UI。
- 不宣称两局样本能够证明 Elo 不下降；需要正式棋力保证时，再增加局面集回归或自对弈 SPRT。
