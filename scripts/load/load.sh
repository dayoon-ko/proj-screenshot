#!/bin/zsh
# Fixed foreground load for the "busy" timing condition:
#   (1) Speedometer 3.1 served from a local copy (no network variance), auto-started with 1000 iterations, in a separate Chrome
#       instance with its own profile (the user's Chrome profile and windows are untouched);
#   (2) a 1080p30 H.264 video (Big Buck Bunny, Blender Foundation, CC-BY) looping in QuickTime Player.
# usage: ./load.sh start | stop | status
# Assets (not in the repo; put them next to this script):
#   Speedometer-release-3.1/   https://github.com/WebKit/Speedometer (tag release/3.1), the repository root
#   bbb_sunflower_1080p_30fps_normal.mp4   https://download.blender.org/demo/movies/BBB/ (CC-BY Blender Foundation)
HERE=${0:A:h}; PORT=${LOAD_PORT:-8765}; PROF="$HERE/.chrome_profile"; VIDEO="$HERE/bbb_sunflower_1080p_30fps_normal.mp4"
SPEEDO="http://localhost:$PORT/index.html?startAutomatically&iterationCount=1000"
case "$1" in
start)
  [ -d "$HERE/Speedometer-release-3.1" ] && [ -f "$VIDEO" ] || { echo "load assets missing, see the header of $0"; exit 1; }
  /usr/bin/python3 -m http.server $PORT --bind 127.0.0.1 --directory "$HERE/Speedometer-release-3.1" > "$HERE/http.log" 2>&1 &
  echo $! > "$HERE/.http.pid"; sleep 1
  open -na "Google Chrome" --args --user-data-dir="$PROF" --no-first-run --no-default-browser-check --disable-background-timer-throttling \
       --window-position=0,0 --window-size=1300,950 "$SPEEDO"
  osascript -e "tell application \"QuickTime Player\"" -e "set d to open POSIX file \"$VIDEO\"" -e "set looping of d to true" \
            -e "set muted of d to true" -e "play d" -e "end tell" > /dev/null
  nohup "$HERE/video_watchdog.sh" >> "$HERE/video_watchdog.log" 2>&1 &
  echo "load started $(date '+%F %T')";;
stop)
  pkill -f "user-data-dir=$PROF"
  osascript -e "tell application \"QuickTime Player\" to close (every document whose name contains \"bbb_sunflower\") saving no" > /dev/null 2>&1
  [ -f "$HERE/.http.pid" ] && kill "$(cat "$HERE/.http.pid")" 2>/dev/null; rm -f "$HERE/.http.pid"
  echo "load stopped $(date '+%F %T')";;
status)
  echo "chrome(load): $(pgrep -f "user-data-dir=$PROF" | wc -l | tr -d ' ') procs | http: $(pgrep -f "http.server $PORT" | wc -l | tr -d ' ') | quicktime docs:"
  osascript -e 'tell application "QuickTime Player" to get name of every document' 2>/dev/null;;
*) echo "usage: $0 start|stop|status"; exit 1;;
esac
