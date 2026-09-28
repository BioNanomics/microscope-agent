# Adaptive time-lapse: proposal and status

## The question behind this

In the long time-lapse, the specimen pulled back from one region at about
the 25 hour mark, and it looked instantaneous. Two readings are possible:

- **Biology.** One cell, no nutrients in that region, so it withdraws its
  material and redirects it. A retraction moves mass toward the main body,
  so the connecting tubes and body should thicken in the frames just after.
- **Filming artefact.** A stalled acquisition, a focus loss, an
  illumination change, or a stage bump can all make a region "vanish"
  between two frames.

The rendered video cannot settle this. It may drop or duplicate frames,
and it carries no timestamps. The raw frames can.

## How to answer it (when at the microscope)

`timelapse/frame_audit.py` scores every consecutive pair of frames in a
sequence and flags, per interval:

| Flag | Meaning |
|---|---|
| `GAP x3.2` | the interval was 3.2x the median: the acquisition stalled, the change is a missing stretch of time |
| `INTENSITY +25%` | the whole field got brighter or darker: illumination or exposure, not the specimen |
| `SHIFT 12px` | the whole field translated: stage or sample moved |
| `CHANGE` | the specimen's footprint or pixels changed with none of the above: biology |

Run it on the frames around the event with the real timestamps:

```
python -m timelapse.frame_audit RAW_FRAME_DIR --timestamps frame_times.csv --around 25h --window 1h
```

`frame_times.csv` is one timestamp per frame, in frame order, exported from
the raw dataset's metadata. Without it the tool falls back to file
modification times, which are only trustworthy if the files were written
as they were captured.

**Finer time slices exist only if the raw dataset has more frames than
the video.** Compare the two frame counts first. If the run was a fixed
interval and the video used every frame, there is nothing finer to
recover, and the honest answer is that the interval was too coarse.

## The proposal: image slowly, look cheaply, image fast when it matters

The fixed-interval recorder on this branch,
`acquisition/orchestration/timelapse.py`, stays as it is: a steady 10 s
series is the right tool for the shuttle-streaming rhythm, and its
drift-free scheduling is reused here. The adaptive loop is for the long,
slow runs where the interesting minutes are rare and unpredictable.

A fixed interval spends light, disk and time evenly on the boring hours
and the interesting minutes alike. The adaptive loop puts the fast
frames where the change is:

1. **Slow mode.** One capture per slow interval, previewed at low
   resolution.
2. **Cheap change score, no model.** Each frame is compared with a
   running median of the last few frames: pixel change relative to the
   baseline's own noise, change in the specimen's footprint area, and a
   whole-field shift check. This is numpy on a 256 pixel thumbnail and
   costs nothing per frame, so it can run for days.
3. **Burst mode.** When the score crosses the threshold, capture at the
   fast interval for a fixed window. Further change inside the window
   extends it. When the window lapses, return to slow.
4. **Model only at the trigger.** A hook fires once per trigger, not
   once per frame. That is where a vision model looks at the before and
   after frames and says "extend", "ignore", or where a notification
   goes out. This keeps model calls rare and gives a written reason per
   event.
5. **Shifts never start a burst.** A whole-field translation is the
   stage or the sample moving. It is logged as its own event. Burst
   imaging a drifted field is wasted light.

### Safety and budget

- **Hard caps.** Maximum captures and maximum runtime end the run in any
  mode.
- **One approval per run, not per frame.** The unattended loop cannot
  stop for a human on every capture. On real hardware the whole plan
  (intervals, burst length, caps, thresholds) is printed and approved
  once at the terminal before the first frame. The scheduler then
  supplies the per-call hardware confirmation itself. No model ever
  supplies it.
- **Light budget.** For brightfield the slow loop is nearly free. For
  fluorescence the slow loop adds phototoxicity, so the preview channel
  should be the least damaging one, and the burst window should be
  short.
- **Nothing else moves.** The loop only captures. It does not move the
  stage, refocus, or change exposure. Those stay under the existing
  human-gated tools.

### What is built (branch `adaptive-timelapse`)

- `get_image(backend="mock")`: a simulated capture with no camera, so all
  of this runs off the microscope PC. The real path is unchanged and
  still gated.
- `timelapse/change_detector.py`: the change score.
- `timelapse/frame_audit.py`: the audit tool above.
- `timelapse/scheduler.py`: the slow/burst loop, with the caps, the
  one-time approval, and the trigger hook.
- `timelapse/model_trigger.py`: Claude behind that hook. At each trigger
  it sees the frame before, the frame that fired, and the detector's
  numbers, and answers "extend" or "ignore" with a one-sentence reason
  that is logged. Capped calls per run, never more than one ask per 30 s,
  and every failure (no key, no network, refusal, odd reply) is a
  harmless "no opinion". Enabled with `--model-trigger`. Verified live on
  the mock on 2026-09-28: at a synthetic shrink event the model answered
  "extend" and gave the right reason (specimen contracted, no field shift
  or illumination change). The call runs on a background thread: a
  second live run showed burst spacing held exactly while the model took
  9 s to answer. A reply that lands after its burst has ended is logged
  as late and changes nothing. Verified against the mock:
  swapping the served frame mid-run started a burst within one slow
  interval and returned to slow afterwards.
- A test suite (24 tests, synthetic frames with known answers) and a CI
  workflow that runs it on every push.

Try it on any machine:

```
python -m timelapse.scheduler --backend mock --slow 2 --burst 0.5 --burst-duration 5 --max-runtime 30
```

### What is next

1. **Tune thresholds on a real recording.** Run the audit on the existing
   time-lapse. The scores at the known 25 hour event and during quiet
   stretches give the starting thresholds for the scheduler.
2. **First real run, attended.** Short slow interval, low caps, someone
   watching, brightfield only.
3. **Decide on focus.** Long runs drift. Either the Perfect Focus System
   holds it, or the loop needs a bounded refocus step, which would be a
   new, gated capability.

### One request

Export the per-frame timestamps from the raw 25 hour dataset, or just
report the raw frame count next to the video frame count. That settles
whether finer slices exist before anyone spends time looking for them.
