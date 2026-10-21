#!/bin/zsh
# Keeps the load video playing while the load Chrome instance is alive; logs every re-open with a timestamp (video_watchdog.log).
HERE=${0:A:h}; PROF="$HERE/.chrome_profile"; VIDEO="$HERE/bbb_sunflower_1080p_30fps_normal.mp4"
while pgrep -f "user-data-dir=$PROF" > /dev/null; do
  st=$(osascript -e 'tell application "QuickTime Player" to get playing of (every document whose name contains "bbb_sunflower")' 2>/dev/null)
  if [[ "$st" != *true* ]]; then
    echo "$(date '+%F %T') video not playing (state='$st') -> reopen"
    osascript -e "tell application \"QuickTime Player\"" -e "set d to open POSIX file \"$VIDEO\"" -e "set looping of d to true" \
              -e "set muted of d to true" -e "play d" -e "end tell" > /dev/null 2>&1
  fi
  sleep 10
done
echo "$(date '+%F %T') load Chrome gone -> watchdog exits"
