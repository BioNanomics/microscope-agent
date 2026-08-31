# Removing the EAA integration

This integration was built to be fully, cleanly removable. To remove it
completely:

## 1. Delete this folder

```
rm -rf eaa_integration/
```

That's it for code/config/dependencies - `eaa-core` and everything it
pulled in only ever lived in `eaa_integration/.venv/`, which this
deletes. No package was ever installed into the main project's
`.venv/`, and no line inside `eaa_integration/` references anything
outside this folder except a read-only path to the main project's
Python executable and `mcp_server/server_loop.py` (never modified).

## 2. Revert the two touch points outside this folder

Both are additive-only (new lines, nothing rewritten) and safe to
revert independently of each other:

- **`.gitignore`** - remove these three lines (added for this
  integration's runtime files):
  ```
  eaa_integration/memory/
  eaa_integration/*.sqlite
  ```
  (The `.venv/` and `.env` lines already existed before this
  integration and are still needed by the rest of the project - leave
  those alone.)

- **Nothing else.** `mcp_server/server_loop.py`, `mcp_server/loop_tools.py`,
  `requirements.txt`, `README.md`, and every other file outside this
  folder were never modified to build this integration - there is
  nothing else to revert.

## 3. Confirm nothing's left over

```
git status --short
```

Should show only the `.gitignore` change from step 2 (or nothing, if
you've already committed the removal). No other file in the repo
should be touched.
