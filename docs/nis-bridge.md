# NIS bridge: driving the microscope through NIS-Elements

The rest of this repo reaches the hardware directly: the Ti2 stand through
its ActiveX SDK and the Baumer camera through GenICam. Some things only
NIS-Elements can do, above all the AX confocal lasers and the ND
experiments set up in its ND Acquisition window. The NIS bridge makes
those available to our code and to an MCP client.

**Status: status, optical-config listing, XY moves, objective changes and capture work on the scope (2026-10-29). Capture runs on NIS's main thread and NIS no longer crashes. ND runs (steps 7-9) are not yet tested.**
The test plan below is what still has to pass on the scope. Driving
capture through NIS reverses an earlier team decision (see the header of
`mcp_server/baumer_ti2/baumer_ti2_tools.py`), so it also needs sign-off before it is used
for real work.

## How it fits together

```
MCP client (Claude)
   │  stdio
   ▼
mcp_server/nis/nis_server.py          confirm gate, capture-folder rule, estop tool
   │
acquisition/backends/nis_bridge.py   e-stop checked before sending
   │  HTTP, 127.0.0.1:8766
   ▼
acquisition/nis_bridge/bridge_job.py  JOBS Python task, runs inside nis_ar.exe
   │  ctypes
   ▼
g5_regprocs.dll  (NIS macro functions: StgMoveXY, Stg_SetNosepiecePosition, Capture, ND_RunExperiment, ...)
```

The MCP server stays in this repo's environment. Only `bridge_job.py`
runs inside NIS, and it needs nothing beyond the Python standard library,
so nothing has to be installed into NIS's own Python. If our code fails,
NIS keeps running.

`baumer_ti2_server.py` and its tools are unchanged. The NIS tools are a
separate server, `nis-mcp`.

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
5. Stop it with `python -m acquisition.backends.nis_bridge shutdown`.
   **Do not use the JOBS Abort button**: a Python task cannot be interrupted
   by it and NIS freezes (seen 2026-10-14; `limjob.RunContext` has no abort
   flag). The job also ends by itself after 4 h with no request.

## Live monitor

`python -m acquisition.nis_monitor` in a second terminal shows the bridge
state (stage, Z, objective, ND run, e-stop) above a scrolling list of every
action sent through the bridge, from any process, so you can watch what
Claude does. Actions are logged to `logs/nis/events-<date>.jsonl` as a
`start` line and then `ok`, `failed`, `refused` or `error`. While one long
call (an ND run, a capture) holds the bridge, the monitor shows BUSY and
names the action and how long it has run. Use it while working through the
test plan below.

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
| 11 | MCP | Add `nis-mcp` to the MCP client config and ask for `nis_status` | Same answer as step 1. Motion tools refuse without `confirm=True`. |

If steps 7–8 show that `ND_RunExperiment` blocks, the bridge can still run
experiments, but nothing else can be asked while one is in progress. The
fix is to start runs in a way that returns at once. Record the result
before changing anything.

## Results on the microscope (2026-10-13)

- Steps 0-4 and 11 passed. The nosepiece index is **0-based** (4x is 0).
- **Step 5 failed with the first design.** `ChangeObjective` only switches NIS's
  calibration label; the turret did not move (Ti2 SDK still read position 1).
  `Stg_SetNosepiecePosition` alone turns the turret but leaves NIS's objective
  name and calibration on the old objective (2026-10-14), so the bridge now
  turns the turret, waits for it, then calls `ChangeObjective`, and reports
  `arrived` once position and name both match. **Re-run step 5 with the new job.**
- Steps 6-10 not run yet.

## Results on the microscope (2026-10-15)

- **Step 0 failed.** With the first design NIS did not respond while the job
  ran. A JOBS Python task runs on NIS's main thread, and the serving loop
  never returned. JOBS has no background option (the Python task has no
  options besides its code; `ctx` is "currently not implemented"), and
  NIS exposes no outside automation interface we could find, so the loop
  cannot be moved out of NIS by configuration.
- **Change:** `run()` now starts `serve()` on a worker thread and returns, so
  the job finishes at once and NIS stays free. **Unverified:** whether NIS
  macro calls work from a non-main thread. Re-run the plan from step 0, then
  `status` (read-only), then the moves, and stop at the first odd result.
  If calls fail or misbehave off the main thread, the fallback is a short job
  that handles one queued request per run, started from a JOBS loop or timer.

## Results on the microscope (2026-10-15, worker-thread bridge)

- Step 0 passes: NIS stays responsive; the job window closes at once while
  the bridge keeps serving on a worker thread. Reads work off the main thread.
- Steps 1-4 pass: `move 100 0` and back (X 137.5 -> 237.5 -> 137.5), a
  1500 um move refused, and the e-stop refused in both the client and, via
  `curl`, the bridge. Moves from the worker thread work.
- **Step 5 passes.** Turret turned 4x -> 10x -> 4x (nosepiece 0 -> 1 -> 0),
  `arrived: true` each way. NIS's objective display is the blue text in the
  bottom status bar and it follows: `PLAN APO lambdaD 4x OFN25 (8.67 um/px @ 512 x 512)`
  and `PLAN APO lambdaD 10x OFN25 DIC N1 (1.72 um/px @ 512 x 512)`; the Ti2 Pad
  highlights the matching nosepiece slot. **Open question:** 8.67 / 1.72 is
  about 5x, not the 2.5x the magnifications imply. Probably the AX scan
  field differs per objective, but check it with a capture at each.
- Steps 6-10 not run yet.

## Results on the microscope (2026-10-16): capture crashes NIS

- Step 6 (`capture`) killed NIS: `nis_ar.exe` faulted in `Qt5Core.dll`
  (exception 0xc0000409) at the moment of the call. `Capture`,
  `ImageSaveAs` and `CloseCurrentDocument` open and close NIS windows, which
  Qt only allows on the UI thread. No file was saved. Reads, moves and
  objective changes from the worker thread are fine.
- **Fix (untested on the scope):** `run()` executes on NIS's main thread, so it
  registers a Win32 thread timer there (`MainThread.install()` in
  `bridge_job.py`). NIS's own message loop calls the timer every 50 ms, and
  the callback runs the queued `Capture`/`ImageSaveAs`/`CloseCurrentDocument`
  and ND calls on the UI thread. Status, moves and objective changes stay on
  the worker thread. A queued call that the main thread has not started
  within 10 min is cancelled, not run late, and the e-stop is checked again
  on the main thread just before the call starts.
- **To verify:** step 6 (`capture`) first. If NIS still faults, or the timer
  never fires (the call reports "did not run"), the hook is wrong and the
  calls stay refused until it is fixed. If `Capture` blocks NIS's UI for the
  exposure, that is expected. Steps 7-10 (ND) follow only after step 6 passes.
- The bridge job prints "no main-thread hook" and refuses these calls if the
  timer cannot be registered.

## Results on the microscope (2026-10-29): main-thread hook

Bridge job pasted fresh into NIS (JOBS job `NISMCPTest`); its log showed
`NIS bridge v1 listening on 127.0.0.1:8766, pid 28168` and no "no
main-thread hook" warning. Calls went through the MCP tool functions in
`mcp_server/nis/nis_tools.py`, in-process, with `confirm=True` where needed.
Checks on the NIS screen were made by the operator.

| Step | Call | Time | Result | Evidence |
|---|---|---|---|---|
| 1 | `nis_status` | 18:15:39 | **pass** | XY (0, 0), Z 3594.22 and the 20x matched NIS; turret 0-based (0 = 4x … 5 = S Plan Fluor 20xC) |
| - | `nis_list_optical_configs` | 18:16:59 | **pass** | 14 configs, same as NIS's saved list |
| 2 | `nis_move_relative(100, 0)`, `(-100, 0)` | 20:20:41, 20:21:21 | **pass** | Repeated with a pause: NIS showed X 0.100 mm, then 0.000 mm; Y unchanged. Event log `move ok`, 0.1 s each |
| 5 | `nis_change_objective` 20x → 4x → 10x | 20:21:49, 20:23:30 | **pass** | Nosepiece 2 → 0 → 1, `arrived: true` both times, NIS status bar followed |
| 6 | `nis_capture("bridge_check_2026-10-29.nd2")` | 21:05:03 → 21:05:44 | **pass** | **NIS did not crash**; event log `capture ok`, 41.3 s; file saved (6.5 MB, 3 × 1024 × 1024, GFP/RFP/TD); `nis_status` answered afterwards, NIS responding |
| 7-9 | ND run, status during run, finish | - | **not run** | Needs the template's settings first |

Notes:

- The laser was off for the capture, so the image is dark (max 62 of 65535).
  The active optical configuration was a GFP/RFP/TD one, not DIC_10x: the
  bridge cannot read or set it, so select it in NIS before a real capture.
- `Capture`, `ImageSaveAs` and `CloseCurrentDocument` all returned 0 although
  the file was saved, so 0 is not a failure code for these calls.
- `ChangeObjective` (`rc_calibration`) returned 0 both times, but NIS's
  objective name had already followed the turret. **Open question:** the
  capture's pixel size is 0.215 µm, small for the 10x at 1024 × 1024 unless a
  scan zoom was set. Check whether NIS saved the 10x calibration.
- Activating an AX optical configuration in NIS can turn the turret to that
  configuration's objective.

## Safety rules (enforced in code)

- Motion and acquisition answer **POST** only, so a browser cannot trigger them.
- The **e-stop** is checked in the client and again in the bridge. Ending an
  ND run is always allowed.
- XY moves are **relative, at most 1000 µm per axis** per call. **Z is never moved.**
- Objective changes are limited to **4× and 10×**, the long working distances.
- MCP tools that move or acquire need **`confirm=True`**.
- `nis_capture` saves only into the capture folder
  (`CONFOCAL_NIS_CAPTURE_DIR`, else `data/NIS/captures`; ND experiments save wherever their "Save to File" path points, use `data/NIS/nd_runs`), by plain file name,
  and never overwrites.
- The bridge listens on **127.0.0.1** only.
