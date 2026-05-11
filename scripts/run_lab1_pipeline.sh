#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FORCE=0
DRY_RUN=0
SKIP_YOLO=0
SELECTED_TASKS="task1,task2,task3"

usage() {
  cat <<'EOF' >&2
Usage: bash ./scripts/run_lab1_pipeline.sh [TASKS...] [--force] [--dry-run] [--skip-yolo]

Arguments:
  TASKS                 Task names to run (default: task1,task2,task3)
                        Valid values: task1, task2, task3
                        Example: run_lab1_pipeline.sh task2 task3

Options:
  --force     Force rerun all tasks even if completed
  --dry-run   Print commands without executing
  --skip-yolo Skip YOLO mask generation in task3

Examples:
  # Run all tasks
  bash ./scripts/run_lab1_pipeline.sh

  # Run only task2 and task3
  bash ./scripts/run_lab1_pipeline.sh task2 task3

  # Run only task1 with dry-run
  bash ./scripts/run_lab1_pipeline.sh task1 --dry-run

EOF
  exit 1
}

# Parse tasks from positional arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-yolo)
      SKIP_YOLO=1
      shift
      ;;
    task1|task2|task3)
      # Accumulate task names
      if [[ -n "$SELECTED_TASKS" ]]; then
        SELECTED_TASKS="${SELECTED_TASKS},$1"
      else
        SELECTED_TASKS="$1"
      fi
      shift
      ;;
    *)
      usage
      ;;
  esac
done

# Convert comma-separated list to array
IFS=',' read -ra selected_tasks_array <<< "$SELECTED_TASKS"
valid_tasks=(task1 task2 task3)

# Validate tasks
for task in "${selected_tasks_array[@]}"; do
  if [[ ! " ${valid_tasks[*]} " =~ " $task " ]]; then
    echo "Invalid task: $task" >&2
    echo "Valid tasks: ${valid_tasks[*]}" >&2
    exit 1
  fi
done

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   Camera3D Complete Lab1 Pipeline Runner    ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Tasks:     ${SELECTED_TASKS}"
echo "  Force:     $FORCE"
echo "  Dry Run:    $DRY_RUN"
echo "  Skip YOLO: $SKIP_YOLO"
echo ""

# Task definitions
declare -A TASKS
TASKS[task1_fps_sweep]="Task1 FPS sweep (4, 8, 16, 30 fps) for S1-1, S1-2, S1-3|scripts/task1_fps_sweep_full.sh"
TASKS[task2_full]="Task2 with optimized default subsequences (return_mid, scan_stable, return_long)|scripts/task2_full_pipeline.sh"
TASKS[task3_full]="Task3 with raw, default, motion, yolo masks|scripts/task3_full_pipeline.sh"

# Function to check if task output exists
check_task1_complete() {
  local fps_list=(4 8 16 30)
  local videos=(S1-1 S1-2 S1-3)
  for video in "${videos[@]}"; do
    for fps in "${fps_list[@]}"; do
      local case_root="outputs/lab1/task1/${video}_fps${fps}"
      local required_files=(
        "${case_root}/sparse/0/images.txt"
        "${case_root}/sparse/0/points3D.txt"
        "${case_root}/trajectory.png"
        "${case_root}/sparse_points.png"
      )
      for file in "${required_files[@]}"; do
        [[ -f "$file" ]] || return 1
      done
    done
  done
  return 0
}

check_task2_complete() {
  [[ -f "outputs/lab1/task2/S1-2_fps30/summary.csv" ]]
}

check_task3_complete() {
  local videos=(S2-1 S2-2)
  for video in "${videos[@]}"; do
    local case_root="outputs/lab1/task3/${video}_fps30"
    local raw_sparse="${case_root}/raw/sparse/0/images.txt"
    local mask_sparse="${case_root}/mask_default/sparse/0/images.txt"
    [[ -f "$raw_sparse" ]] || return 1
    [[ -f "$mask_sparse" ]] || return 1
  done
  return 0
}

# Function to run a task
run_task() {
  local task_name="$1"
  local task_desc="$2"
  local task_script="$3"

  echo ""
  echo "========================================"
  echo "Task: $task_name"
  echo "Description: $task_desc"
  echo "========================================"
  echo ""

  # Check if already completed
  if [[ $FORCE -eq 0 ]]; then
    case "$task_name" in
      task1_fps_sweep)
        check_task1_complete && { echo "✓ Task already completed. Use --force to rerun."; return 0; }
        ;;
      task2_full)
        check_task2_complete && { echo "✓ Task already completed. Use --force to rerun."; return 0; }
        ;;
      task3_full)
        check_task3_complete && { echo "✓ Task already completed. Use --force to rerun."; return 0; }
        ;;
    esac
  fi

  # Build command
  local cmd=("bash" "$task_script")
  [[ $FORCE -eq 1 ]] && cmd+=("--force")
  [[ $DRY_RUN -eq 1 ]] && cmd+=("--dry-run")
  [[ $task_name == "task3_full" && $SKIP_YOLO -eq 1 ]] && cmd+=("--skip-yolo")

  # Display command
  echo "Running: $task_script ${cmd[*]:1}"
  echo ""

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY RUN] Would execute task: $task_name"
    return 0
  fi

  # Execute
  if "${cmd[@]}"; then
    echo "✓ Task completed successfully"
    return 0
  else
    echo "✗ Task failed"
    return 1
  fi
}

# Main execution
failed_tasks=()
success_count=0

for task_name in "${selected_tasks_array[@]}"; do
  # Map task name to script path
  case "$task_name" in
    task1)
      script="scripts/task1_fps_sweep_full.sh"
      desc="Task1 FPS sweep (4, 8, 16, 30 fps) for S1-1, S1-2, S1-3"
      ;;
    task2)
      script="scripts/task2_full_pipeline.sh"
      desc="Task2 with optimized default subsequences (return_mid, scan_stable, return_long)"
      ;;
    task3)
      script="scripts/task3_full_pipeline.sh"
      desc="Task3 with raw, default, motion, yolo masks"
      ;;
  esac

  if run_task "$task_name" "$desc" "$script"; then
    ((success_count++))
  else
    failed_tasks+=("$task_name")
  fi
done

# Summary
echo ""
echo "========================================"
echo "Pipeline Execution Summary"
echo "========================================"
echo ""

echo "Total tasks: ${#selected_tasks_array[@]}"
echo "  Successful: $success_count"
echo "  Failed:    ${#failed_tasks[@]}"

if [[ ${#failed_tasks[@]} -gt 0 ]]; then
  echo ""
  echo "Failed tasks:"
  for task_name in "${failed_tasks[@]}"; do
    echo "  - $task_name"
  done
  exit 1
fi

echo ""
echo "✓ All tasks completed successfully!"
echo ""
echo "Output locations:"
echo "  Task1: outputs/lab1/task1/"
echo "  Task2: outputs/lab1/task2/"
echo "  Task3: outputs/lab1/task3/"
