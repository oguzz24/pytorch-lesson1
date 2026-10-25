import gc
import time
import torch


SIZES = [
    (64, 1024, 1024),
    (128, 512, 512),
    (256, 256, 256),
]


OPERATIONS = {
    "Matrix multiplication": lambda a, b: torch.matmul(a, b),
    "Addition": lambda a, b: a + b,
    "Multiplication": lambda a, b: a * b,
    "Transpose": lambda a, b: a.transpose(-1, -2).contiguous(),
    "Sum": lambda a, b: a.sum(),
}


def validate_tensors(a, b):
    """Check tensor shapes and data types."""
    if a.shape != b.shape:
        raise ValueError("Tensor shapes must be equal.")

    if not a.is_floating_point() or not b.is_floating_point():
        raise TypeError("Tensors must contain floating point values.")


def measure_cpu(operation, a, b):
    """Measure CPU execution time."""
    validate_tensors(a, b)

    start = time.time()
    result = operation(a, b)
    end = time.time()

    del result

    return (end - start) * 1000


def measure_gpu(operation, a, b):
    """Measure CUDA execution time."""
    validate_tensors(a, b)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    result = operation(a, b)
    end.record()

    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end)

    del result

    return elapsed


def benchmark_cpu(size):
    """Run all operations on CPU."""
    a = torch.rand(size, dtype=torch.float32)
    b = torch.rand(size, dtype=torch.float32)

    results = {}

    for name, operation in OPERATIONS.items():
        results[name] = measure_cpu(operation, a, b)

    del a
    del b

    gc.collect()

    return results


def benchmark_gpu(size):
    """Run all operations on CUDA."""
    a = torch.rand(size, dtype=torch.float32, device="cuda")
    b = torch.rand(size, dtype=torch.float32, device="cuda")

    results = {}

    for name, operation in OPERATIONS.items():
        operation(a, b)
        torch.cuda.synchronize()

        results[name] = measure_gpu(operation, a, b)

    del a
    del b

    torch.cuda.empty_cache()

    return results


def print_results(size, cpu_results, gpu_results):
    """Print benchmark results in table format."""
    print(f"\nTensor size: {size}")
    print("-" * 82)

    print(
        f"{'Operation':<25}"
        f"{'CPU (ms)':>15}"
        f"{'GPU (ms)':>15}"
        f"{'Speedup':>15}"
    )

    print("-" * 82)

    for name in OPERATIONS:
        cpu_time = cpu_results[name]

        if gpu_results is not None:
            gpu_time = gpu_results[name]

            if gpu_time > 0:
                speedup = cpu_time / gpu_time
            else:
                speedup = 0.0

            print(
                f"{name:<25}"
                f"{cpu_time:>15.3f}"
                f"{gpu_time:>15.3f}"
                f"{speedup:>14.2f}x"
            )
        else:
            print(
                f"{name:<25}"
                f"{cpu_time:>15.3f}"
                f"{'N/A':>15}"
                f"{'N/A':>15}"
            )


def analyze_results(all_results):
    """Analyze CPU and CUDA benchmark results."""
    print("\nAnalysis")

    if torch.cuda.is_available():
        best_operation = None
        best_speedup = 0.0
        best_size = None

        for size, cpu_results, gpu_results in all_results:
            if gpu_results is None:
                continue

            for name in OPERATIONS:
                gpu_time = gpu_results[name]

                if gpu_time > 0:
                    speedup = cpu_results[name] / gpu_time

                    if speedup > best_speedup:
                        best_speedup = speedup
                        best_operation = name
                        best_size = size

        if best_operation is not None:
            print(
                f"Highest measured GPU speedup: {best_operation}, "
                f"{best_speedup:.2f}x, tensor size {best_size}"
            )
    else:
        print(
            "CUDA is not available, so GPU speedup cannot be measured "
            "on this computer."
        )

    print(
        "Operations with many parallel calculations can receive greater "
        "speedup on a GPU."
    )

    print(
        "Some operations can be slower on a GPU because launching GPU "
        "operations and transferring data introduce overhead."
    )

    print(
        "As tensor size increases, GPU parallel processing can become more "
        "effective compared with CPU execution."
    )

    print(
        "Transferring data between CPU and GPU takes additional time and can "
        "reduce the overall performance benefit."
    )


def run_tests():
    """Run simple tests for benchmark operations."""
    a = torch.rand(2, 3, 3)
    b = torch.rand(2, 3, 3)

    assert torch.matmul(a, b).shape == (2, 3, 3)
    assert (a + b).shape == a.shape
    assert (a * b).shape == a.shape
    assert a.transpose(-1, -2).shape == (2, 3, 3)
    assert a.sum().numel() == 1

    print("\nAll tests passed successfully.")


def main():
    """Run CPU and CUDA performance comparisons."""
    cuda_available = torch.cuda.is_available()

    print("CUDA available:", cuda_available)

    if cuda_available:
        print("GPU:", torch.cuda.get_device_name(0))

    all_results = []

    for size in SIZES:
        print(f"\nTesting {size}...")

        try:
            cpu_results = benchmark_cpu(size)
        except (RuntimeError, MemoryError) as error:
            print("CPU test failed:", error)
            continue

        gpu_results = None

        if cuda_available:
            try:
                gpu_results = benchmark_gpu(size)
            except RuntimeError as error:
                print("GPU test failed:", error)
                torch.cuda.empty_cache()

        print_results(size, cpu_results, gpu_results)

        all_results.append(
            (size, cpu_results, gpu_results)
        )

    analyze_results(all_results)
    run_tests()


if __name__ == "__main__":
    main()