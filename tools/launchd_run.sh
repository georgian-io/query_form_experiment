#!/bin/bash
# launchd_run.sh <label> <job-script.sh>
# Runs <job-script> under launchd — reparented to PID 1, out of Claude Code's process tree.
# KeepAlive={SuccessfulExit:false}: auto-restarts on kill/crash (job must be RESUMABLE), stops on
# exit 0 (job signals genuine completion with exit 0). Logs (appended across restarts) to LOG.
LABEL="$1"; JOB="$2"
[ -z "$LABEL" ] || [ -z "$JOB" ] && { echo "usage: launchd_run.sh <label> <job-script>"; exit 1; }
JOB="$(cd "$(dirname "$JOB")" && pwd)/$(basename "$JOB")"
LOG="$HOME/.claude/launchd-logs/${LABEL}.log"; mkdir -p "$(dirname "$LOG")"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>${LABEL}</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>${JOB}</string></array>
  <key>WorkingDirectory</key><string>$(pwd)</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>StandardOutPath</key><string>${LOG}</string>
  <key>StandardErrorPath</key><string>${LOG}</string>
</dict></plist>
PL
launchctl unload "$PLIST" 2>/dev/null
launchctl load -w "$PLIST" && echo "running '${LABEL}' under launchd | log: ${LOG}"
echo "stop/clean: launchctl unload '${PLIST}' && rm -f '${PLIST}'"
