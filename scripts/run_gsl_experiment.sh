#!/usr/bin/env bash
# 명령어 하나로 GSL 실험을 실행하고 로그+bag을 기록한다.
#
# 사용법:
#   run_gsl_experiment.sh <mode> [gui]
#     <mode> : spiral      = SPIRAL만 (baseline)
#              integrated  = Kernel DM + SPIRAL 통합
#     [gui]  : 생략        = headless (화면 없음, 모바일/SSH 장시간 수집용)
#              gui         = Gazebo + RViz 창 표시 (눈으로 확인/시연)
#
# 예:
#   run_gsl_experiment.sh spiral            # SPIRAL만, 화면 없음
#   run_gsl_experiment.sh integrated gui    # 통합, 화면 표시
#
# 기록물은 ~/gsl_runs/ 아래에 모드+타임스탬프로 저장된다:
#   ~/gsl_runs/<label>_<timestamp>.log        (전체 콘솔 로그, RESPAWN/FOUND 라인 포함)
#   ~/gsl_runs/bag_<label>_<timestamp>/       (분석용 토픽 rosbag)
#
# 중지: Ctrl-C (launch, bag record 모두 함께 종료됨)
set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  spiral)     USE_GRID_REFINE=false; MODE_LABEL="spiral_only"     ;;
  integrated) USE_GRID_REFINE=true;  MODE_LABEL="kerneldm_spiral" ;;
  *)
    echo "사용법: $0 <mode> [gui]" >&2
    echo "  <mode> spiral     : SPIRAL만 사용 (baseline)   -> 파일명 spiral_only_*" >&2
    echo "         integrated : Kernel DM + SPIRAL 통합    -> 파일명 kerneldm_spiral_*" >&2
    echo "  [gui]  생략=headless(화면없음) / gui=Gazebo+RViz 창 표시" >&2
    exit 1 ;;
esac

# 두 번째 인자로 GUI 여부 결정 (기본: headless)
GUI="${2:-}"
case "$GUI" in
  gui)  LAUNCH_FILE="cleanroom_spawn.launch.py";    DISPLAY_MODE="GUI(Gazebo+RViz)" ;;
  "")   LAUNCH_FILE="cleanroom_headless.launch.py"; DISPLAY_MODE="headless(화면없음)" ;;
  *)
    echo "두 번째 인자는 'gui' 또는 생략만 가능합니다 (입력값: '$GUI')" >&2
    exit 1 ;;
esac

# ROS 환경 소싱 (이미 소싱돼 있어도 안전)
# ROS setup.bash들이 미설정 변수를 참조하므로 소싱 구간만 nounset(-u)을 잠시 끈다
set +u
source /opt/ros/humble/setup.bash
source "$HOME/icir_ws/install/setup.bash"
set -u

RUN_DIR="$HOME/gsl_runs"
mkdir -p "$RUN_DIR"

# 이미 실행 중인 이전 실험이 있으면: bag을 먼저 정상 종료(flush)해 데이터를 보존한 뒤
# 시뮬레이터/launch를 정리하고 나서 새 실험을 시작한다. (포트/노드이름 충돌 방지)
stop_existing_runs() {
  local bag_pids launch_pids
  bag_pids=$(pgrep -f "ros2 bag record -o $RUN_DIR/bag_" || true)
  launch_pids=$(pgrep -f "ros2 launch icir_cleanroom cleanroom_" || true)

  if [[ -z "$bag_pids" && -z "$launch_pids" ]] \
     && ! pgrep -x gzserver >/dev/null; then
    return 0   # 실행 중인 이전 실험 없음
  fi

  echo "[run_gsl_experiment] 이전 실험이 실행 중입니다 - 정상 종료 후 재시작합니다."

  # 1) bag record에 SIGINT -> 기록을 flush/close (데이터 손실 방지). 완전히 닫힐 때까지 대기.
  if [[ -n "$bag_pids" ]]; then
    echo "  - 이전 bag 기록 정상 종료(flush) 중..."
    kill -INT $bag_pids 2>/dev/null || true
    for _ in $(seq 1 40); do   # 최대 20초 대기
      pgrep -f "ros2 bag record -o $RUN_DIR/bag_" >/dev/null || break
      sleep 0.5
    done
    if pgrep -f "ros2 bag record -o $RUN_DIR/bag_" >/dev/null; then
      echo "  - bag이 제때 닫히지 않아 한 번 더 종료 신호 전송..."
      pkill -INT -f "ros2 bag record -o $RUN_DIR/bag_" 2>/dev/null || true
      sleep 3
    fi
  fi

  # 2) 이전 launch에 SIGINT -> nav2/gazebo 정상 종료 유도
  if [[ -n "$launch_pids" ]]; then
    echo "  - 이전 시뮬레이션 종료 중..."
    kill -INT $launch_pids 2>/dev/null || true
    sleep 4
  fi

  # 3) 남은 프로세스 강제 정리 (bag은 위에서 이미 닫혔으므로 여기서 -9 해도 안전)
  pkill -9 -f "ros2 launch icir_cleanroom cleanroom_" 2>/dev/null || true
  pkill -9 -x gzserver 2>/dev/null || true
  pkill -9 -x gzclient 2>/dev/null || true
  pkill -9 -f "gas_patrol_node.py" 2>/dev/null || true
  pkill -9 -f "component_container" 2>/dev/null || true
  sleep 2
  echo "  - 이전 실험 정리 완료. 새 실험을 시작합니다."
  echo ""
}

stop_existing_runs
TS="$(date +%Y%m%d_%H%M%S)"
# 파일명에 모드를 명시해 gsl_runs 폴더에서 바로 구분 가능하게 함
#   spiral_only_*      : SPIRAL 단독
#   kerneldm_spiral_*  : Kernel DM + SPIRAL 통합
LOG_FILE="$RUN_DIR/${MODE_LABEL}_${TS}.log"
BAG_DIR="$RUN_DIR/bag_${MODE_LABEL}_${TS}"

echo "=========================================="
echo " GSL 실험 시작"
echo "  모드       : $MODE_LABEL (use_grid_refine=$USE_GRID_REFINE)"
echo "  표시        : $DISPLAY_MODE"
echo "  로그       : $LOG_FILE"
echo "  bag        : $BAG_DIR"
echo "  중지       : Ctrl-C"
echo "=========================================="

# 분석에 필요한 토픽만 기록 (장시간 실행 대비 용량 관리)
BAG_TOPICS=(
  /gas_source/pose                  # ground truth 소스 위치
  /gas_source/concentration         # 소스 강도
  /gas_sensor/detected_concentration
  /gas_source/estimated_pose        # 그리드 argmax 추정 위치
  /gas_source/estimated_heatmap
  /odom                             # 로봇 위치
)

# 종료 시 자식 프로세스(launch, bag) 정리
cleanup() {
  echo ""
  echo "[run_gsl_experiment] 종료 - 자식 프로세스 정리 중..."
  [[ -n "${BAG_PID:-}" ]] && kill "$BAG_PID" 2>/dev/null || true
  [[ -n "${LAUNCH_PID:-}" ]] && kill "$LAUNCH_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  echo "[run_gsl_experiment] 정리 완료. 로그: $LOG_FILE"
}
trap cleanup INT TERM EXIT

# bag 기록 시작 (백그라운드)
ros2 bag record -o "$BAG_DIR" "${BAG_TOPICS[@]}" \
  > "$RUN_DIR/bag_${MODE_LABEL}_${TS}.record.log" 2>&1 &
BAG_PID=$!

# launch 실행 - 콘솔 출력을 화면과 로그 파일 양쪽으로 (tee)
stdbuf -oL -eL ros2 launch icir_cleanroom "$LAUNCH_FILE" \
  use_grid_refine:="$USE_GRID_REFINE" 2>&1 | tee "$LOG_FILE" &
LAUNCH_PID=$!

wait "$LAUNCH_PID"
