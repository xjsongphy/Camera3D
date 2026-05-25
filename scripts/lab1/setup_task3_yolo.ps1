param(
    [string]$Model = "yolo11s-seg.pt"
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $RootDir

Write-Host "Sync uv environment with YOLO extra..."
uv sync --extra task3-yolo

Write-Host "Preparing local model cache: models/$Model"
New-Item -ItemType Directory -Force -Path "models" | Out-Null

$py = @'
from pathlib import Path
from ultralytics import YOLO

model_name = "__MODEL__"
model_path = Path("models") / model_name

print(f"Loading model: {model_name}")
model = YOLO(model_name)

source = Path(model.ckpt_path) if getattr(model, "ckpt_path", None) else None
if source and source.exists() and source.resolve() != model_path.resolve():
    model_path.write_bytes(source.read_bytes())
    print(f"Saved local checkpoint: {model_path}")
else:
    print("Checkpoint was already local or unavailable for copy.")
'@

$py = $py.Replace("__MODEL__", $Model)
$py | python -

Write-Host "YOLO setup finished."
Write-Host "Use model with: --model models/$Model"
