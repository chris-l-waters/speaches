# Speaches - Parallel Speech-to-Text Processing

A high-performance OpenAI-compatible speech-to-text server with advanced parallel processing capabilities. This fork is focused on improving CPU-only performance for workflows that require transcription of many short audio files.

## Overview

Speaches provides a containerized speech-to-text API that supports true parallel processing within a single container instance. The implementation uses ProcessPoolExecutor to bypass Python's Global Interpreter Lock (GIL), enabling multiple simultaneous transcription jobs to run on separate CPU cores.

## Parallel Processing Architecture

### Implementation Details

- **Executor**: ProcessPoolExecutor (bypasses Python GIL)
- **CPU Affinity**: Each worker process bound to specific CPU cores
- **Model Instances**: Isolated per process to avoid threading conflicts
- **Memory**: Higher usage due to process isolation but better performance
- **Fallback**: ThreadPoolExecutor available for compatibility

### Configuration

Parallel processing is controlled through environment variables:

```yaml
environment:
  - WHISPER__MODEL_INSTANCES_PER_MODEL=4    # Number of model instances per model
  - WHISPER__MAX_CONCURRENT_JOBS=4          # Maximum parallel workers
  - WHISPER__ENABLE_CPU_AFFINITY=true       # Bind workers to CPU cores
  - WHISPER__USE_PROCESS_POOL=true          # Use ProcessPoolExecutor (recommended)
  # Threading environment optimizations
  - OMP_NUM_THREADS=1
  - OPENBLAS_NUM_THREADS=1
  - MKL_NUM_THREADS=1
  - NUMBA_NUM_THREADS=1
```

## Performance Benchmarks

Comprehensive benchmarking was performed using identical audio files processed sequentially vs. in parallel:

### Performance Results

| Configuration | Workers | Mean Time | Speedup | Parallel Factor | Efficiency |
|---------------|---------|-----------|---------|-----------------|------------|
| Single-threaded (Baseline) | 1       | 0.180s    | 1.00x   | 1.00            | 100.0%     |
| Parallel 2 Workers | 2       | 0.105s    | 1.72x   | 1.61            | 86.2%      |
| Parallel 3 Workers | 3       | 0.148s    | 1.22x   | 2.14            | 40.8%      |
| Parallel 4 Workers | 4       | 0.186s    | 0.97x   | 2.66            | 24.2%      |

### Performance Analysis

- **Best Configuration**: Parallel 2 Workers
- **Maximum Speedup**: 1.72x
- **Optimal Efficiency**: 86.2% with 2 workers
- **GIL Bypass**: ✅ Complete (separate processes)

**Key Insights:**
- 2 workers provide the optimal balance of performance and resource efficiency
- Beyond 2 workers, process overhead begins to outweigh parallel benefits for typical workloads
- True parallel processing is achieved through process isolation
- CPU affinity ensures optimal core utilization

### Technical Implementation

The parallel processing system consists of three main components:

#### 1. Model Pool Management
```python
class ModelPool:
    """Pool of identical model instances for parallel processing."""
```
- Manages multiple instances of the same model
- Uses threading.Queue for thread-safe instance allocation
- Supports dynamic scaling based on configuration

#### 2. CPU Affinity Executor
```python
class CPUAffinityThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that assigns specific CPU cores to each worker thread."""
```
- Binds worker threads/processes to specific CPU cores
- Reduces context switching and improves cache locality
- Configurable core assignment strategy

#### 3. Process-Based Transcription
```python
def _process_transcribe(model_id: str, audio_data, worker_id: int, whisper_config_dict: dict, **kwargs):
    """Process-based transcription function that runs in a separate process to bypass GIL."""
```
- Completely separate process for each transcription job
- Bypasses Python's Global Interpreter Lock entirely
- Isolated memory space prevents model interference

## Usage

### Docker Compose

```yaml
services:
  speaches:
    container_name: speaches
    build: .
    ports:
      - 8000:8000
    environment:
      - WHISPER__MODEL_INSTANCES_PER_MODEL=2
      - WHISPER__MAX_CONCURRENT_JOBS=2
      - WHISPER__ENABLE_CPU_AFFINITY=true
      - WHISPER__USE_PROCESS_POOL=true
      - OMP_NUM_THREADS=1
      - OPENBLAS_NUM_THREADS=1
      - MKL_NUM_THREADS=1
      - NUMBA_NUM_THREADS=1
```

### API Usage

The parallel processing is transparent to the API client:

```bash
# Single transcription
curl -X POST \
  -F "file=@audio.wav" \
  -F "model=Systran/faster-distil-whisper-small.en" \
  -F "response_format=json" \
  http://localhost:8000/v1/audio/transcriptions

# Multiple parallel requests are automatically handled
```

### Enabling/Disabling Parallel Processing

You can control parallel processing per request:

```bash
# Force parallel processing
curl -X POST \
  -F "file=@audio.wav" \
  -F "model=Systran/faster-distil-whisper-small.en" \
  -F "use_parallel=true" \
  http://localhost:8000/v1/audio/transcriptions

# Force single-threaded processing
curl -X POST \
  -F "file=@audio.wav" \
  -F "model=Systran/faster-distil-whisper-small.en" \
  -F "use_parallel=false" \
  http://localhost:8000/v1/audio/transcriptions
```

## Architecture Deep Dive

### Problem: Python's Global Interpreter Lock (GIL)

Python's GIL prevents true parallel execution of CPU-bound tasks within a single process. Traditional threading approaches resulted in minimal performance gains:

- **Threading**: 13-15% improvement (limited by GIL)
- **Model Contention**: Multiple threads competing for single model instance
- **CPU Underutilization**: All threads running on single core

### Solution: Process-Based Parallelism

Our implementation uses ProcessPoolExecutor to achieve true parallelism:

1. **Separate Processes**: Each transcription runs in isolated process
2. **GIL Bypass**: No shared Python interpreter state
3. **CPU Affinity**: Process binding to specific CPU cores
4. **Model Isolation**: Each process loads its own model instance

### Resource Management

#### Memory Usage
- **Higher Memory**: Each process loads separate model instance (~300MB per model)
- **Isolation**: No memory conflicts between concurrent jobs
- **Scalability**: Memory usage scales linearly with worker count

#### CPU Utilization
- **Core Binding**: Workers assigned to specific CPU cores
- **Reduced Context Switching**: Processes stay on assigned cores
- **Cache Locality**: Better CPU cache utilization

#### I/O Handling
- **Async API**: FastAPI handles concurrent HTTP requests
- **Process Delegation**: CPU-bound work delegated to separate processes
- **Non-blocking**: Main process remains responsive

## Configuration Reference

### Environment Variables

| Variable | Description | Default | Range |
|----------|-------------|---------|-------|
| `WHISPER__MODEL_INSTANCES_PER_MODEL` | Number of model instances per model | 1 | 1-8 |
| `WHISPER__MAX_CONCURRENT_JOBS` | Maximum parallel workers | 4 | 1-8 |
| `WHISPER__ENABLE_CPU_AFFINITY` | Enable CPU core binding | `true` | boolean |
| `WHISPER__USE_PROCESS_POOL` | Use ProcessPoolExecutor vs ThreadPoolExecutor | `true` | boolean |

### Threading Environment Variables

To prevent library-level threading conflicts:

```yaml
environment:
  - OMP_NUM_THREADS=1           # OpenMP threading
  - OPENBLAS_NUM_THREADS=1      # BLAS operations
  - MKL_NUM_THREADS=1           # Intel MKL
  - NUMBA_NUM_THREADS=1         # Numba JIT compilation
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Reduce `MODEL_INSTANCES_PER_MODEL` 
   - Use ThreadPoolExecutor instead: `USE_PROCESS_POOL=false`

2. **CPU Affinity Errors**
   - Disable affinity: `ENABLE_CPU_AFFINITY=false`
   - Check container CPU limits

3. **Performance Degradation**
   - Optimal worker count is typically 2-4
   - Consider workload size vs overhead

### Monitoring

Check container logs for parallel processing activity:

```bash
docker logs speaches | grep -E "(ProcessPool|Worker|Process worker)"
```

Expected log entries:
```
Using ProcessPoolExecutor with 4 workers for GIL bypass
Process worker 0: bound to CPU cores [0, 1, 2, 3]
Process worker 1: transcribed audio in 0.15 seconds
```

## Benchmarking

Run your own benchmarks using the included test scripts:

```bash
# Comprehensive benchmark for README table
python readme_benchmark.py

# Quick performance test
python quick_test.py

# Statistical analysis with confidence intervals
python statistical_benchmark.py
```

## Original Features

All original speaches features are preserved:

- OpenAI API compatibility
- Streaming transcription support  
- Text-to-Speech with Kokoro and Piper
- Dynamic model loading/offloading
- GPU and CPU support
- Realtime API
- Docker deployment

For complete feature documentation, visit [speaches.ai](https://speaches.ai/).

## To-Do: Future Performance Improvements

### Phase 2: Advanced Optimizations

- **GPU Process Pools**: Extend ProcessPoolExecutor to support GPU-bound models with CUDA process isolation
- **Dynamic Worker Scaling**: Implement auto-scaling based on queue depth and system load
- **Memory-Mapped Models**: Investigate shared memory approaches to reduce per-process memory overhead
- **Async Model Loading**: Pipeline model loading with transcription to reduce cold-start latency

### Phase 3: Distributed Processing

- **Multi-Container Orchestration**: Scale across multiple container instances with load balancing
- **Redis-Based Job Queue**: Replace in-memory queues with distributed Redis queues for horizontal scaling
- **Model Sharding**: Split large models across multiple processes/containers for memory efficiency
- **Streaming Optimizations**: Implement chunked processing for real-time streaming scenarios

### Phase 4: Hardware-Specific Optimizations

- **NUMA Awareness**: Optimize memory allocation and CPU affinity for multi-socket systems
- **Intel MKL-DNN Integration**: Leverage optimized neural network primitives for CPU inference
- **ARM64 Optimizations**: Performance tuning for Apple Silicon and ARM-based cloud instances
- **Custom CUDA Kernels**: Implement specialized kernels for faster-whisper bottlenecks

### Monitoring & Observability

- **Prometheus Metrics**: Expose detailed performance metrics for monitoring
- **Distributed Tracing**: Add OpenTelemetry support for request flow analysis
- **Resource Profiling**: Built-in CPU/memory profiling for optimization insights
- **Benchmarking Suite**: Automated performance regression testing

### Research Areas

- **Model Quantization**: Explore INT8/FP16 quantization for memory and speed improvements
- **Speculative Decoding**: Implement parallel hypothesis generation for faster inference
- **Batch Processing**: Optimize for high-throughput batch transcription scenarios
- **Edge Deployment**: Lightweight variants for resource-constrained environments

*Contributions and suggestions for these improvements are welcome!*

## License

MIT License (same as original project)