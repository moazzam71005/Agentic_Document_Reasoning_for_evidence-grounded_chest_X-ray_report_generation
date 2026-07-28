# Windows Shell UnicodeEncodeError in Print Statements

Link back to: [[notes/context/overview.md|Project Overview]]

## Problem Description
During long-running background evaluations on Windows, printing non-ASCII characters such as the checkmark `✓` (`\u2713`) caused a terminal crash with `UnicodeEncodeError: 'charmap' codec can't encode character '\u2713' in position 0: character maps to <undefined>`.

## Root Cause
Windows console hosts and PowerShell instances default to localized encodings (e.g., `cp1252`) rather than UTF-8. When Python scripts attempt to print Unicode symbols to stdout without a configured encoding environment variable, the default encoding handler fails to map non-ASCII characters, triggering an encoding error.

## Solution & Code Update
We resolved this by:
1. Replacing non-ASCII symbols with clean, standard ASCII prefixes (e.g., `[SUCCESS]`) in the print statements of the script.
2. Setting the `PYTHONIOENCODING` environment variable to `utf-8` during execution in PowerShell:
   ```powershell
   $env:PYTHONIOENCODING="utf-8"
   ```
