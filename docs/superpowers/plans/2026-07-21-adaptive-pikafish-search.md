# Adaptive Pikafish Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable Pikafish's native adaptive time management, prevent CPU oversubscription, and make analysis cancellation stop the active UCI search.

**Architecture:** Keep the C++ engine unchanged. Make the existing Python UCI adapter send both a synthetic clock and the user's hard `movetime`, cap the existing Threads option at eight, and relay the existing UI stop action to the engine process.

**Tech Stack:** Python 3 standard library, `unittest`, Tkinter, Pikafish UCI.

## Global Constraints

- Work only on branch `dev-1.0`.
- Do not modify the nested `pikafish` repository.
- Add no dependency, configuration file, opening book, tablebase, GPU model, or custom search heuristic.
- Preserve the 1–60 second input range and the default 10-second hard limit.
- Keep Hash at 128 MB, MultiPV at 1, and `ucinewgame` before independent snapshot analysis.
- Use test-driven development: each production behavior must first have a test that fails for the expected reason.

---

### Task 1: Adaptive UCI adapter

**Files:**
- Modify: `test_assistant.py:947-985`
- Modify: `pikafish_engine.py:154-222`

**Interfaces:**
- Consumes: `PikafishEngine._configure_strength(uci_lines)` and `PikafishEngine.get_best_move(position, movetime=5000)`.
- Produces: `PikafishEngine.stop_search() -> None`; adaptive UCI command with `wtime`, `btime`, `movestogo`, and hard `movetime`.

- [ ] **Step 1: Write failing engine tests**

Replace the current all-CPU expectation and add adaptive-command and cancellation coverage:

```python
def test_configure_strength_caps_threads_at_eight(self):
    engine = object.__new__(PikafishEngine)
    sent = []
    engine.send_command = sent.append
    uci_lines = [
        "option name Threads type spin default 1 min 1 max 1024",
        "option name Hash type spin default 16 min 1 max 33554432",
        "option name MultiPV type spin default 1 min 1 max 128",
    ]

    with patch("pikafish_engine.os.cpu_count", return_value=12):
        engine._configure_strength(uci_lines)

    self.assertEqual(sent[0], "setoption name Threads value 8")
    self.assertEqual(sent[1:], [
        "setoption name Hash value 128",
        "setoption name MultiPV value 1",
    ])

def test_engine_uses_adaptive_time_with_hard_limit(self):
    engine = object.__new__(PikafishEngine)
    engine._lock = threading.Lock()
    sent = []
    engine.send_command = sent.append
    engine._read_until = lambda _predicate, _timeout: ["bestmove a0a1"]
    command = f"position fen {board_to_fen(STANDARD_BOARD, 'w')} moves h0g2"

    engine.get_best_move(command, movetime=5000)

    self.assertEqual(sent, [
        "ucinewgame",
        "isready",
        command,
        "go wtime 10000 btime 10000 movestogo 10 movetime 5000",
    ])

def test_stop_search_sends_stop_only_to_running_process(self):
    engine = object.__new__(PikafishEngine)
    process = Mock()
    process.poll.return_value = None
    engine.process = process

    engine.stop_search()

    process.stdin.write.assert_called_once_with("stop\n")
    process.stdin.flush.assert_called_once_with()

    process.stdin.reset_mock()
    process.poll.return_value = 0
    engine.stop_search()
    process.stdin.write.assert_not_called()
```

- [ ] **Step 2: Run engine tests and verify RED**

Run:

```text
python3 -m unittest -v \
  test_assistant.EngineTests.test_configure_strength_caps_threads_at_eight \
  test_assistant.EngineTests.test_engine_uses_adaptive_time_with_hard_limit \
  test_assistant.EngineTests.test_stop_search_sends_stop_only_to_running_process
```

Expected: the thread and UCI command assertions fail against current behavior, and `stop_search` is missing.

- [ ] **Step 3: Implement the minimum adapter change**

In `PikafishEngine._configure_strength()` cap threads without adding configuration:

```python
threads = min(
    maximum("Threads", 1),
    8,
    max(1, os.cpu_count() or 1),
)
```

In `get_best_move()` replace the fixed command with:

```python
movetime = int(movetime)
clock = movetime * 2
self.send_command(
    f"go wtime {clock} btime {clock} movestogo 10 movetime {movetime}"
)
```

Add the best-effort cancellation method without taking the search lock:

```python
def stop_search(self):
    process = self.process
    if not process or process.poll() is not None:
        return
    try:
        process.stdin.write("stop\n")
        process.stdin.flush()
    except (BrokenPipeError, OSError):
        pass
```

- [ ] **Step 4: Run engine tests and verify GREEN**

Run the three-test command from Step 2. Expected: all three tests pass.

- [ ] **Step 5: Commit Task 1**

```text
git add pikafish_engine.py test_assistant.py
git commit -m "perf: enable adaptive Pikafish search"
```

### Task 2: Relay UI cancellation

**Files:**
- Modify: `test_assistant.py:1125-1145`
- Modify: `app.py:181-184`
- Modify: `app.py:536-541`

**Interfaces:**
- Consumes: `PikafishEngine.stop_search() -> None` from Task 1.
- Produces: `AssistantApp.stop()` that sets `stop_event`, notifies the current engine, disables the stop button, and reports stopping status.

- [ ] **Step 1: Write the failing UI cancellation test**

Add:

```python
def test_stop_notifies_running_engine(self):
    app = object.__new__(AssistantApp)
    app.stop_event = threading.Event()
    app.engine_lock = threading.Lock()
    app.engine = Mock()
    app.start_button = Mock()
    app.status_var = Mock()

    app.stop()

    self.assertTrue(app.stop_event.is_set())
    app.engine.stop_search.assert_called_once_with()
    app.start_button.configure.assert_called_once_with(state="disabled")
    app.status_var.set.assert_called_once_with("正在停止分析")
```

- [ ] **Step 2: Run the UI test and verify RED**

Run:

```text
python3 -m unittest -v test_assistant.LoopTests.test_stop_notifies_running_engine
```

Expected: failure because current `AssistantApp.stop()` does not call `engine.stop_search()`.

- [ ] **Step 3: Implement relay and label change**

Change the UI label to `最长秒`. Update `AssistantApp.stop()` to:

```python
def stop(self):
    self.stop_event.set()
    with self.engine_lock:
        engine = self.engine
    if engine:
        engine.stop_search()
    self.start_button.configure(state="disabled")
    self.status_var.set("正在停止分析")
```

- [ ] **Step 4: Run UI and engine tests**

Run:

```text
python3 -m unittest -v \
  test_assistant.LoopTests.test_stop_notifies_running_engine \
  test_assistant.LoopTests.test_analyze_snapshot_drops_result_after_stop \
  test_assistant.EngineTests.test_stop_search_sends_stop_only_to_running_process
```

Expected: all three tests pass.

- [ ] **Step 5: Commit Task 2**

```text
git add app.py test_assistant.py
git commit -m "fix: stop active Pikafish analysis"
```

### Task 3: Documentation and full verification

**Files:**
- Modify: `README.md:18-34`

**Interfaces:**
- Consumes: final adaptive-search and cancellation behavior from Tasks 1 and 2.
- Produces: user documentation matching the UI label, 10-second hard-limit semantics, eight-thread cap, and 128 MB Hash.

- [ ] **Step 1: Update README behavior description**

Replace the fixed-time/all-CPU/512-MB wording with:

```text
“最长秒”默认是 10，可设置 1～60 秒。它是单次搜索的绝对上限；Pikafish 会根据最佳着稳定度和局面变化提前结束，当前测试机器通常约 2～5 秒返回。实际耗时取决于局面和硬件。程序最多使用 8 个逻辑 CPU、128 MB 哈希和单主变化线。
```

Update the later fixed-search sentence to say that the result is the best move found before adaptive completion or the selected hard limit.

- [ ] **Step 2: Run complete automated verification**

Run:

```text
python3 -m unittest -v test_assistant.py
python3 -m py_compile app.py capture.py board_recognition.py pikafish_engine.py game_state.py
git diff --check
```

Expected: 90 existing tests plus new tests pass with one existing screenshot test skipped; compilation and whitespace checks exit zero.

- [ ] **Step 3: Run real-engine adaptive and cancellation checks**

Use `PikafishEngine` against the bundled binary to verify:

```python
engine = PikafishEngine()
move, score = engine.get_best_move(board_to_fen(STANDARD_BOARD, "w"), 2000)
assert re.fullmatch(r"[a-i][0-9][a-i][0-9]", move)
engine.close()
```

Then start a 10-second search in a worker thread, call `stop_search()` after the search begins, and assert the worker exits well before the hard limit.

- [ ] **Step 4: Inspect final repository state and commit docs**

```text
git status --short
git diff --stat HEAD~2
git add README.md
git commit -m "docs: explain adaptive search limit"
```

Expected: only the approved adapter, UI, tests, and README changes exist; the nested `pikafish` repository remains unchanged.
