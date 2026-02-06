from marker.utils.gpu import GPUManager

# Default CUDA batch sizes (from builders/processors)
_DEFAULT_CUDA_BATCH_SIZES = {
    "layout_batch_size": 12,
    "detection_batch_size": 10,
    "table_rec_batch_size": 14,
    "ocr_error_batch_size": 14,
    "recognition_batch_size": 48,
    "equation_batch_size": 32,
}

# Observed peak VRAM with default batch sizes (~10GB)
_DEFAULT_PEAK_VRAM_GB = 10


def get_batch_sizes_worker_counts(gpu_manager: GPUManager, peak_worker_vram: int):
    vram = gpu_manager.get_gpu_vram()

    workers = max(1, vram // peak_worker_vram)
    if workers == 1:
        return {}, workers

    # Scale batch sizes proportionally to the target per-worker VRAM budget
    scale = peak_worker_vram / _DEFAULT_PEAK_VRAM_GB

    batch_sizes = {}
    for key, default_val in _DEFAULT_CUDA_BATCH_SIZES.items():
        batch_sizes[key] = max(2, int(default_val * scale))

    batch_sizes["detector_postprocessing_cpu_workers"] = 2

    return batch_sizes, workers
