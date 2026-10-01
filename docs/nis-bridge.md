# NIS bridge: driving the microscope through NIS-Elements

The rest of this repo reaches the hardware directly: the Ti2 stand through
its ActiveX SDK and the Baumer camera through GenICam. Some things only
NIS-Elements can do, above all the AX confocal lasers and the ND
experiments set up in its ND Acquisition window. The NIS bridge makes
those available to our code and to an MCP client.

**Status: built and tested off the microscope, not yet run inside NIS.**
The test plan below is what still has to pass on the scope. Driving
capture through NIS reverses an earlier team decision (see the header of
`mcp_server/loop_tools.py`), so it also needs sign-off before it is used
for real work.

## How it fits together

```
MCP client (Claude)
   │  stdio
   ▼
mcp_server/server_nis.py          confirm gate, capture-folder rule, estop tool
   │
acquisition/backends/nis_bridge.py   e-stop checked before sending
   │  HTTP, 127.0.0.1:8766
   ▼
acquisition/nis_bridge/bridge_job.py  JOBS Python task, runs inside nis_ar.exe
   │  ctypes
   ▼
g5_regprocs.dll  (NIS macro functions: StgMoveXY, ChangeObjective, Capture, ND_RunExperiment, ...)
```

The MCP server stays in this repo's environment. Only `bridge_job.py`
runs inside NIS, and it needs nothing beyond the Python standard library,
so nothing has to be installed into NIS's own Python. If our code fails,
NIS keeps running.

`server_loop.py` and its tools are unchanged. The NIS tools are a
separate server, `confocal-mcp-nis`.

## What was already verified (2026-09-29)

- A JOBS Python task runs inside `nis_ar.exe` itself and can bind a port
  that another process on the PC reaches (`nis_port_probe.py`).
- `g5_regprocs.dll` exports the macro functions by name. NIS's own
  `limpy.macro` calls `WaitText` this way. Signatures come from the macro
  reference in `C:\Program Files\NIS-Elements\Docs\nis\eng_ar`.

## Setting up the bridge in NIS

1. **Start conditions.** The Ti2 controller is on, NIS-Elements was started
   after it, and the e-stop is clear (`python -m acquisition.estop status`).
2. In NIS, open **JOBS** and create a job with one **Python** task.
3. Replace the task's code with the whole of
   `acquisition/nis_bridge/bridge_job.py`, then save the job, for example
   as `NIS bridge`.
4. Run the job. Its log shows
   `NIS bridge v1 listening on 127.0.0.1:8766, pid <NIS pid>`.
5. The bridge runs until you **Abort** the job, or until
   `python -m acquisition.backends.nis_bridge shutdown`.

## Test plan for the microscope

Run each step from a terminal in the repo (`.venv\Scripts\python -m
acquisition.backends.nis_bridge ...`). Write down the output. Stop at the
first surprise. Press the STOP panel if anything moves unexpectedly.

| # | Step | Command | Pass if |
|---|------|---------|---------|
| 0 | NIS usable while the job runs | click around NIS for 30 s | NIS does not freeze. **If it does, stop here**: JOBS runs Python on the main thread, and the bridge needs a different host. |
| 1 | Read-only status | `status` | `xy_um` and `z_um` match the Ti2 Pad (Z ≈ 4600, not 1.0). `nosepiece_objectives` lists the turret. **Note whether position 1 is the 4×**, i.e. whether the index is 0- or 1-based. |
| 2 | Small move | `move 100 0`, then `move -100 0` | `after_um` changes by +100 then −100 in X, and the Ti2 Pad shows the same. The sign of X on the image is recorded. |
| 3 | Move cap | `move 1500 0` | Refused, and nothing moves. |
| 4 | E-stop | `python -m acquisition.estop engage`, then `move 10 0`, then release | Refused by the client. Then send the same move with `curl -X POST -d "{\"dx\":10}" http://127.0.0.1:8766/move` and check the bridge refuses it too. |
| 5 | Objective | `objective "<exact 10× name from status>"`, then back to the 4× | **Does the nosepiece physically turn?** `nosepiece_before/after` go 1 → 2 → 1, and NIS's objective display follows. If only the calibration changes, the bridge needs a nosepiece call instead. |
| 6 | Capture | `capture D:\QurratulAin_ConfocalOrchestratorProject\CelegansAML-18\bridge_check_01.nd2` | The file exists and opens with GFP/RFP/TD (check with `python -m analysis.aml18_survey <file>`). No stray image window is left open in NIS. |
| 7 | Saved ND experiment | In ND Acquisition, set a **2-loop, 1-minute** time-lapse with Save to File and **Save** it as `bridge test`. Then `nd-run "bridge test"`. | The run starts and the file appears. **Record `call_returned_after_s`**: ~0 means the call returns at once, ~60 means it blocks until the run ends. |
| 8 | Status during a run | during step 7, from a second terminal: `status` | Answers with `nd_running: true`, or hangs until the run ends (the blocking case from step 7). |
| 9 | Finish early | run `bridge test` with more loops, then `nd-finish` | The run ends after the current loop, and the file keeps the frames so far. |
| 10 | Shutdown | `shutdown` | The job ends by itself, and `status` then reports the bridge unreachable. |
| 11 | MCP | Add `confocal-mcp-nis` to the MCP client config and ask for `nis_status` | Same answer as step 1. Motion tools refuse without `confirm=True`. |

If steps 7–8 show that `ND_RunExperiment` blocks, the bridge can still run
experiments, but nothing else can be asked while one is in progress. The
fix is to start runs in a way that returns at once. Record the result
before changing anything.

## Safety rules (enforced in code)

- Motion and acquisition answer **POST** only, so a browser cannot trigger them.
- The **e-stop** is checked in the client and again in the bridge. Ending an
  ND run is always allowed.
- XY moves are **relative, at most 1000 µm per axis** per call. **Z is never moved.**
- Objective changes are limited to **4× and 10×**, the long working distances.
- MCP tools that move or acquire need **`confirm=True`**.
- `nis_capture` saves only into the capture folder
  (`CONFOCAL_NIS_CAPTURE_DIR`, else `data/nis_captures`), by plain file name,
  and never overwrites.
- The bridge listens on **127.0.0.1** only.
