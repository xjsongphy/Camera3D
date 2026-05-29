"""
结构光训练脚本 — 配置文件驱动。

用法:
    # 指定配置文件
    uv run python scripts/lab2/run_training.py --config configs/lab2/sl_marble_objects.yaml

    # 命令行覆盖个别参数
    uv run python scripts/lab2/run_training.py --config configs/lab2/default.yaml --iterations 500 --decoder zncc_nn

输出:
    results/lab2/{timestamp}_{scene}_{decoder}_{penalty}/
        config.json              训练配置快照
        training_log.csv         iteration, loss, elapsed_sec
        patterns_initial.pt      初始图案
        patterns_final.pt        最终图案
        patterns_final.png       图案可视化
        spectrum_final.png       频谱图
        loss_curve.png           损失曲线
        depth_gt.png             GT 深度图
        gt_corr.png              GT 对应关系图
        rendered_final.png       最终渲染图
        correspondence_final.png 预测 vs GT 对应关系
        depth_error_map.png      对应误差热力图
        metrics_final.json       MAE, RMSE, acc_0, acc_1, acc_2

    当 decoder=both 时额外生成对比目录:
    results/lab2/{timestamp}_{scene}_comparison_{penalty}/
        comparison_loss_curves.png    损失曲线叠加对比
        comparison_correspondence.png 对应关系并排对比
        comparison_error_maps.png     误差图并排对比
        comparison_patterns.png       图案并排对比
        comparison_metrics.json       两个 decoder 的指标
        comparison_metrics.png        指标柱状图对比
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# 确保 src/ 在 import 路径上
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from lab2.OpticalSGD import DecoderType, OptimizerConfig, OpticalSGDOptimizer
from lab2.decoder import correspondence_metrics, hard_decode
from lab2.scene_genertor import (
    SCENE_PRESETS,
    create_standard_renderer,
    get_standard_camera_config,
    get_standard_projector_config,
)


# =============================================================================
# 工具函数
# =============================================================================


def auto_detect_device(device_str: str) -> torch.device:
    """自动检测设备。'auto' → 优先 CUDA，否则 CPU。"""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件。"""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_output_dir(base_dir: str, scene: str, decoder: str, penalty: str) -> Path:
    """创建带时间戳的输出目录（在 runs/ 子目录下）。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{timestamp}_{scene}_{decoder}_{penalty}"
    output_dir = Path(base_dir) / "runs" / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# =============================================================================
# 场景初始化
# =============================================================================


def init_renderer(scene_name: str, device: torch.device, spp: int):
    """
    创建渲染器并用真实 Mitsuba 深度替换合成默认深度。

    create_standard_renderer() 内部调用 load_scene()，无参数时生成合成深度。
    此处强制调用 render_depth_for_visualization() 获取真实深度。
    """
    renderer = create_standard_renderer(scene_name, device=str(device), spp=spp)

    # 强制覆盖合成深度为真实 Mitsuba 深度
    renderer._depth = None
    renderer._gt_corr = None
    renderer.render_depth_for_visualization()

    return renderer


# =============================================================================
# 可视化 & 保存
# =============================================================================


def save_config_snapshot(output_dir: Path, cfg: dict) -> None:
    """保存训练配置快照。"""
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False, default=str)


def _save_checkpoint(
    optimizer: OpticalSGDOptimizer,
    log_rows: list[dict],
    ckpt_dir: Path,
) -> None:
    """
    保存完整 checkpoint：patterns + decoder 参数 + optimizer 状态 + 训练日志。
    可用于 --resume 续训。
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "iteration": optimizer.iteration,
        "patterns": optimizer.patterns.detach().cpu(),
        "patterns_grad": optimizer.patterns.grad.detach().cpu() if optimizer.patterns.grad is not None else None,
        "decoder_state_dict": optimizer.decoder.state_dict() if optimizer.decoder is not None else None,
        "optimizer_state_dict": optimizer.optimizer.state_dict() if optimizer.optimizer is not None else None,
        "loss_history": optimizer.loss_history,
    }
    torch.save(state, ckpt_dir / "checkpoint.pt")

    # 同时保存训练日志
    if log_rows:
        save_training_log(log_rows, ckpt_dir.parent.parent)  # 保存到输出根目录


def _load_checkpoint(ckpt_path: Path, optimizer: OpticalSGDOptimizer) -> int:
    """
    从 checkpoint 恢复训练状态。

    Returns:
        已完成的 iteration 数
    """
    state = torch.load(ckpt_path, map_location=optimizer.renderer.device, weights_only=False)

    # 恢复 patterns
    optimizer.patterns = torch.nn.Parameter(
        state["patterns"].to(device=optimizer.renderer.device, dtype=optimizer.renderer.dtype)
    )

    # 恢复 decoder 参数（ZNCC-NN 有可学习参数）
    if state.get("decoder_state_dict") is not None and optimizer.decoder is not None:
        optimizer.decoder.load_state_dict(state["decoder_state_dict"])

    # 重建 optimizer 并恢复状态
    if optimizer.config.decoder_type == DecoderType.ZNCC_NN:
        optimizer.optimizer = torch.optim.Adam([
            {"params": [optimizer.patterns], "lr": optimizer.config.learning_rate},
            {"params": list(optimizer.decoder.parameters()), "lr": optimizer.config.decoder_learning_rate},
        ])
    else:
        optimizer.optimizer = torch.optim.Adam(
            [optimizer.patterns], lr=optimizer.config.learning_rate
        )

    if state.get("optimizer_state_dict") is not None:
        optimizer.optimizer.load_state_dict(state["optimizer_state_dict"])

    # 恢复训练历史
    optimizer.loss_history = state.get("loss_history", [])
    optimizer.iteration = state.get("iteration", 0)

    # 同步到渲染器
    optimizer.renderer.update_patterns(optimizer.patterns.detach())

    return optimizer.iteration


def render_and_save_gt(renderer, output_dir: Path) -> None:
    """渲染并保存 GT 深度图和对应关系图。"""
    depth = renderer.render_depth_for_visualization().cpu().numpy()
    gt_corr = renderer.gt_corr.cpu().numpy()

    # 深度图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    im1 = ax1.imshow(depth, cmap="plasma")
    ax1.set_title(f"GT Depth ({depth.min():.2f}-{depth.max():.2f}m)")
    plt.colorbar(im1, ax=ax1)
    ax2.hist(depth.flatten(), bins=50, edgecolor="black")
    ax2.set_title("Depth Distribution")
    ax2.set_xlabel("Depth (m)")
    plt.tight_layout()
    fig.savefig(output_dir / "depth_gt.png", dpi=150)
    plt.close(fig)

    # 对应关系图
    fig, ax = plt.subplots(figsize=(8, 4))
    valid = np.isfinite(gt_corr)
    display = np.where(valid, gt_corr, -1)
    im = ax.imshow(display, cmap="viridis")
    ax.set_title("GT Correspondence")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(output_dir / "gt_corr.png", dpi=150)
    plt.close(fig)


def save_training_log(log_rows: list[dict], output_dir: Path) -> None:
    """保存训练日志 CSV。"""
    csv_path = output_dir / "training_log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "loss", "elapsed_sec"])
        writer.writeheader()
        writer.writerows(log_rows)


def save_rendered_images(renderer, patterns: torch.Tensor, output_dir: Path) -> None:
    """用 Mitsuba 渲染最终图案的图像并保存。"""
    K = patterns.shape[0]
    images = renderer.render_images(patterns).detach().cpu().numpy()  # [K, H, W, 3]

    fig, axes = plt.subplots(K, 1, figsize=(12, 3 * K))
    if K == 1:
        axes = [axes]
    for k in range(K):
        img = images[k]
        if img.ndim == 3:
            axes[k].imshow(img)
        else:
            axes[k].imshow(img, cmap="gray")
        axes[k].set_title(f"Pattern {k + 1} Render")
        axes[k].set_xlabel("x")
        axes[k].set_ylabel("y")
    plt.tight_layout()
    fig.savefig(output_dir / "rendered_final.png", dpi=150)
    plt.close(fig)


def evaluate_and_save(renderer, optimizer: OpticalSGDOptimizer, output_dir: Path) -> dict:
    """
    用最终图案评估解码效果，保存对应关系图、误差图和指标。

    Returns:
        评估指标 dict
    """
    patterns = optimizer.patterns
    images = renderer.render_images_autodiff(patterns)
    gt_corr = renderer.gt_corr

    # 解码
    scores = optimizer.decoder(images, patterns)  # [H, W, Wp]
    pred_corr = hard_decode(scores)  # [H, W]

    # 指标
    metrics = correspondence_metrics(pred_corr, gt_corr)
    metrics_dict = {k: v.item() for k, v in metrics.items()}

    # 保存指标
    with open(output_dir / "metrics_final.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    # 对应关系对比图
    pred_np = pred_corr.cpu().numpy()
    gt_np = gt_corr.cpu().numpy()
    valid = np.isfinite(gt_np)
    gt_display = np.where(valid, gt_np, -1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.imshow(pred_np, cmap="viridis")
    ax1.set_title("Predicted Correspondence")
    plt.colorbar(ax1.images[0], ax=ax1)
    ax2.imshow(gt_display, cmap="viridis")
    ax2.set_title("GT Correspondence")
    plt.colorbar(ax2.images[0], ax=ax2)
    plt.tight_layout()
    fig.savefig(output_dir / "correspondence_final.png", dpi=150)
    plt.close(fig)

    # 误差热力图
    err = np.abs(pred_np - gt_np)
    err_display = np.where(valid, err, 0)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.imshow(pred_np, cmap="viridis")
    ax1.set_title("Predicted")
    ax2.imshow(gt_display, cmap="viridis")
    ax2.set_title("Ground Truth")
    im3 = ax3.imshow(err_display, cmap="hot")
    ax3.set_title(f"Error (MAE={metrics_dict['mae']:.2f})")
    plt.colorbar(im3, ax=ax3)
    plt.tight_layout()
    fig.savefig(output_dir / "depth_error_map.png", dpi=150)
    plt.close(fig)

    return metrics_dict


def save_final_visualizations(optimizer: OpticalSGDOptimizer, output_dir: Path) -> None:
    """保存最终的图案、频谱和损失曲线。"""
    for plot_fn, name in [
        (optimizer.plot_patterns, "patterns_final"),
        (optimizer.plot_frequency_spectrum, "spectrum_final"),
        (optimizer.plot_loss_curve, "loss_curve"),
    ]:
        fig = plot_fn()
        fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# 对比图生成（decoder=both 时）
# =============================================================================


def generate_comparison(
    zncc_dir: Path,
    zncc_nn_dir: Path,
    comparison_dir: Path,
) -> None:
    """从两个 decoder 的输出目录生成对比图。"""
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # 加载训练日志
    def load_log(d: Path) -> tuple[list[int], list[float]]:
        rows = []
        with open(d / "training_log.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((int(row["iteration"]), float(row["loss"])))
        iters, losses = zip(*rows) if rows else ([], [])
        return list(iters), list(losses)

    zncc_iters, zncc_losses = load_log(zncc_dir)
    nn_iters, nn_losses = load_log(zncc_nn_dir)

    # 加载指标
    with open(zncc_dir / "metrics_final.json", encoding="utf-8") as f:
        zncc_metrics = json.load(f)
    with open(zncc_nn_dir / "metrics_final.json", encoding="utf-8") as f:
        nn_metrics = json.load(f)

    combined_metrics = {"zncc": zncc_metrics, "zncc_nn": nn_metrics}
    with open(comparison_dir / "comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(combined_metrics, f, indent=2)

    # 1. 损失曲线叠加
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(zncc_iters, zncc_losses, linewidth=2, label="ZNCC")
    ax.plot(nn_iters, nn_losses, linewidth=2, label="ZNCC-NN")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves: ZNCC vs ZNCC-NN")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(comparison_dir / "comparison_loss_curves.png", dpi=150)
    plt.close(fig)

    # 2. 对应关系并排
    zncc_corr_img = plt.imread(zncc_dir / "correspondence_final.png")
    nn_corr_img = plt.imread(zncc_nn_dir / "correspondence_final.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    ax1.imshow(zncc_corr_img)
    ax1.set_title("ZNCC")
    ax1.axis("off")
    ax2.imshow(nn_corr_img)
    ax2.set_title("ZNCC-NN")
    ax2.axis("off")
    plt.tight_layout()
    fig.savefig(comparison_dir / "comparison_correspondence.png", dpi=150)
    plt.close(fig)

    # 3. 误差图并排
    zncc_err_img = plt.imread(zncc_dir / "depth_error_map.png")
    nn_err_img = plt.imread(zncc_nn_dir / "depth_error_map.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    ax1.imshow(zncc_err_img)
    ax1.set_title("ZNCC Error")
    ax1.axis("off")
    ax2.imshow(nn_err_img)
    ax2.set_title("ZNCC-NN Error")
    ax2.axis("off")
    plt.tight_layout()
    fig.savefig(comparison_dir / "comparison_error_maps.png", dpi=150)
    plt.close(fig)

    # 4. 图案并排
    zncc_pat_img = plt.imread(zncc_dir / "patterns_final.png")
    nn_pat_img = plt.imread(zncc_nn_dir / "patterns_final.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    ax1.imshow(zncc_pat_img)
    ax1.set_title("ZNCC Patterns")
    ax1.axis("off")
    ax2.imshow(nn_pat_img)
    ax2.set_title("ZNCC-NN Patterns")
    ax2.axis("off")
    plt.tight_layout()
    fig.savefig(comparison_dir / "comparison_patterns.png", dpi=150)
    plt.close(fig)

    # 5. 指标柱状图
    metric_names = ["mae", "rmse", "acc_0", "acc_1", "acc_2"]
    zncc_vals = [zncc_metrics[m] for m in metric_names]
    nn_vals = [nn_metrics[m] for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, zncc_vals, width, label="ZNCC", alpha=0.8)
    ax.bar(x + width / 2, nn_vals, width, label="ZNCC-NN", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Value")
    ax.set_title("Metrics: ZNCC vs ZNCC-NN")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(comparison_dir / "comparison_metrics.png", dpi=150)
    plt.close(fig)

    print(f"\n对比结果已保存到: {comparison_dir}")
    print(f"  ZNCC     MAE={zncc_metrics['mae']:.3f}  RMSE={zncc_metrics['rmse']:.3f}  acc<1px={zncc_metrics['acc_1']:.3f}")
    print(f"  ZNCC-NN  MAE={nn_metrics['mae']:.3f}  RMSE={nn_metrics['rmse']:.3f}  acc<1px={nn_metrics['acc_1']:.3f}")

    # 创建符号链接（或路径记录文件）指向两个 decoder 的 run 目录
    try:
        comparison_dir.joinpath("zncc_run").symlink_to(zncc_dir, target_is_directory=True)
        comparison_dir.joinpath("zncc_nn_run").symlink_to(zncc_nn_dir, target_is_directory=True)
    except OSError:
        # Windows 上不支持符号链接，用文本文件记录路径
        (comparison_dir / "zncc_run.txt").write_text(str(zncc_dir), encoding="utf-8")
        (comparison_dir / "zncc_nn_run.txt").write_text(str(zncc_nn_dir), encoding="utf-8")


# =============================================================================
# 单个 decoder 训练
# =============================================================================


def run_single_decoder(
    scene_name: str,
    decoder_name: str,
    cfg: dict,
    device: torch.device,
    output_dir: Path | None,
) -> Path:
    """
    运行单个 decoder 的训练。

    Args:
        scene_name: 场景名
        decoder_name: 'zncc' 或 'zncc_nn'
        cfg: 配置 dict
        device: torch 设备
        output_dir: 输出目录（None 时自动生成）

    Returns:
        输出目录路径
    """
    training = cfg["training"]
    rendering = cfg["rendering"]
    output_cfg = cfg["output"]

    decoder_type = DecoderType.ZNCC if decoder_name == "zncc" else DecoderType.ZNCC_NN
    penalty = training.get("penalty", "l1")

    # 自动生成输出目录（如果未指定）
    if output_dir is None:
        base_dir = output_cfg.get("base_dir", "output/lab2")
        output_dir = create_output_dir(base_dir, scene_name, decoder_name, penalty)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  场景: {scene_name}")
    print(f"  Decoder: {decoder_name}")
    print(f"  迭代数: {training['iterations']}")
    print(f"  设备: {device}")
    print(f"  输出: {output_dir}")
    print(f"{'='*60}\n")

    # 保存配置快照
    config_snapshot = {
        "scene": scene_name,
        "decoder": decoder_name,
        "training": training,
        "rendering": rendering,
        "output": output_cfg,
    }
    save_config_snapshot(output_dir, config_snapshot)

    # 创建渲染器（含真实深度）
    renderer = init_renderer(scene_name, device, rendering.get("spp", 64))

    # 保存 GT 深度和对应关系
    render_and_save_gt(renderer, output_dir)

    # 创建优化器配置
    optimizer_config = OptimizerConfig(
        num_iterations=training["iterations"],
        learning_rate=training.get("learning_rate", 0.01),
        decoder_learning_rate=training.get("decoder_lr", 0.01),
        tau=training.get("tau", 50.0),
        decoder_type=decoder_type,
        neighborhood_size=training.get("neighborhood", 1),
        penalty=penalty,
        log_interval=output_cfg.get("log_interval", 10),
        save_interval=output_cfg.get("save_interval", 50),
        output_dir=str(output_dir),
    )

    # 创建优化器
    optimizer = OpticalSGDOptimizer(renderer, optimizer_config)
    projector_width = renderer.projector.width
    num_patterns = training.get("num_patterns", 4)
    init_mode = training.get("init_mode", "random")

    # 初始化图案
    patterns = optimizer.initialize_patterns(
        num_patterns=num_patterns,
        projector_width=projector_width,
        init_mode=init_mode,
    )

    # 保存初始图案
    torch.save(patterns.detach().cpu(), output_dir / "patterns_initial.pt")

    # 从 checkpoint 恢复（如果指定）
    resume_ckpt = cfg.get("resume")
    start_iter = 0
    if resume_ckpt:
        ckpt_path = Path(resume_ckpt)
        if not ckpt_path.exists():
            # 尝试在输出目录下查找
            alt = output_dir / "checkpoint_final" / "checkpoint.pt"
            if alt.exists():
                ckpt_path = alt
            else:
                print(f"  [警告] Checkpoint 未找到: {resume_ckpt}，从头开始训练")
                resume_ckpt = None

        if resume_ckpt:
            completed = _load_checkpoint(ckpt_path, optimizer)
            start_iter = completed
            print(f"  从 checkpoint 恢复: 已完成 {completed} 步")

    # 训练循环
    log_rows = []
    start_time = time.time()
    iterations = training["iterations"]
    log_interval = output_cfg.get("log_interval", 10)
    save_interval = output_cfg.get("save_interval", 50)

    for i in range(start_iter, iterations):
        loss = optimizer.step()
        elapsed = time.time() - start_time
        log_rows.append({"iteration": i + 1, "loss": loss, "elapsed_sec": round(elapsed, 2)})

        if (i + 1) % log_interval == 0 or i == 0:
            print(f"  Iter {i+1}/{iterations}: loss={loss:.6f} ({elapsed:.1f}s)")

        # 周期性保存 checkpoint
        if (i + 1) % save_interval == 0:
            ckpt_dir = output_dir / "checkpoints" / f"iter_{i+1}"
            _save_checkpoint(optimizer, log_rows, ckpt_dir)

    total_time = time.time() - start_time

    # 保存训练日志
    save_training_log(log_rows, output_dir)

    # 保存最终图案
    torch.save(optimizer.patterns.detach().cpu(), output_dir / "patterns_final.pt")

    # 保存最终完整 checkpoint（含 decoder 参数 + optimizer 状态，可续训）
    _save_checkpoint(optimizer, log_rows, output_dir / "checkpoint_final")

    # 保存可视化
    save_final_visualizations(optimizer, output_dir)

    # 评估并保存
    metrics = evaluate_and_save(renderer, optimizer, output_dir)

    # Mitsuba 渲染最终图像（较慢，只跑一次）
    try:
        save_rendered_images(renderer, optimizer.patterns, output_dir)
    except Exception as e:
        print(f"  [警告] Mitsuba 渲染失败: {e}")

    print(f"\n  训练完成: {total_time:.1f}s")
    print(f"  MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  acc<1px={metrics['acc_1']:.3f}")

    return output_dir


# =============================================================================
# 命令行
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="结构光训练脚本 — 配置文件驱动",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  uv run python scripts/lab2/run_training.py --config configs/lab2/sl_marble_objects.yaml\n"
            "  uv run python scripts/lab2/run_training.py --config configs/lab2/default.yaml --iterations 500 --decoder zncc_nn\n"
        ),
    )
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    # 以下参数可覆盖配置文件中的值
    parser.add_argument("--scene", type=str, default=None, help="覆盖场景名")
    parser.add_argument("--decoder", type=str, choices=["zncc", "zncc_nn", "both"], default=None, help="覆盖 decoder")
    parser.add_argument("--iterations", type=int, default=None, help="覆盖迭代数")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default=None, help="覆盖设备")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练（checkpoint.pt 路径，或 'auto' 自动查找最新的）")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)

    # 命令行参数覆盖配置文件
    if args.scene is not None:
        cfg["scene"] = args.scene
    if args.decoder is not None:
        cfg["decoder"] = args.decoder
    if args.iterations is not None:
        cfg.setdefault("training", {})["iterations"] = args.iterations
    if args.device is not None:
        cfg.setdefault("rendering", {})["device"] = args.device
    if args.resume is not None:
        cfg["resume"] = args.resume

    # 验证场景
    scene_name = cfg["scene"]
    if scene_name not in SCENE_PRESETS:
        print(f"错误: 未知场景 '{scene_name}'。可用场景: {list(SCENE_PRESETS.keys())}")
        sys.exit(1)

    # 设备检测
    rendering = cfg.setdefault("rendering", {})
    device = auto_detect_device(rendering.get("device", "auto"))
    print(f"设备: {device}")

    output_cfg = cfg.setdefault("output", {})
    base_dir = output_cfg.get("base_dir", "results/lab2")
    training = cfg.setdefault("training", {})
    penalty = training.get("penalty", "l1")
    decoder = cfg.get("decoder", "zncc")

    if decoder == "both":
        # 运行两个 decoder（各自独立生成 run 目录），然后生成对比
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_dir = Path(base_dir) / "runs" / f"{timestamp}_{scene_name}_comparison_{penalty}"

        # 分别运行两个 decoder，各自生成独立 run 目录
        zncc_dir = run_single_decoder(
            scene_name, "zncc", cfg, device,
            None,  # 自动生成目录
        )
        zncc_nn_dir = run_single_decoder(
            scene_name, "zncc_nn", cfg, device,
            None,  # 自动生成目录
        )

        # 生成对比结果（包含符号链接指向两个 run 目录）
        generate_comparison(zncc_dir, zncc_nn_dir, comparison_dir)
    else:
        output_dir = create_output_dir(base_dir, scene_name, decoder, penalty)
        run_single_decoder(scene_name, decoder, cfg, device, output_dir)


if __name__ == "__main__":
    main()
