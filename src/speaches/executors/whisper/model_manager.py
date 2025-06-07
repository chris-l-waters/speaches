from __future__ import annotations

import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import contextlib
import logging
import os
import queue
import threading
from typing import TYPE_CHECKING

from faster_whisper import WhisperModel

from speaches.model_manager import SelfDisposingModel

if TYPE_CHECKING:
    from speaches.config import (
        WhisperConfig,
    )

logger = logging.getLogger(__name__)


def _process_transcribe(model_id: str, audio_data, worker_id: int, whisper_config_dict: dict, **kwargs):
    """
    Process-based transcription function that runs in a separate process to bypass GIL.
    This function will be executed by ProcessPoolExecutor.
    """
    import os
    import time
    from faster_whisper import WhisperModel
    from speaches.api_types import TranscriptionSegment
    
    # Set CPU affinity for this worker process
    if whisper_config_dict.get('enable_cpu_affinity', False):
        try:
            total_cores = os.cpu_count() or 4
            max_workers = whisper_config_dict.get('max_concurrent_jobs', 4)
            cores_per_worker = max(1, total_cores // max_workers)
            start_core = worker_id * cores_per_worker
            end_core = min(start_core + cores_per_worker, total_cores)
            assigned_cores = list(range(start_core, end_core))
            
            os.sched_setaffinity(0, assigned_cores)
            logger.info(f"Process worker {worker_id}: bound to CPU cores {assigned_cores}")
        except (AttributeError, OSError) as e:
            logger.warning(f"Process worker {worker_id}: CPU affinity failed: {e}")
    
    start_time = time.perf_counter()
    
    try:
        # Create model instance in this process with single-threaded config
        cpu_threads = 1 if whisper_config_dict.get('model_instances_per_model', 1) > 1 else whisper_config_dict.get('cpu_threads', 0)
        num_workers = 1 if whisper_config_dict.get('model_instances_per_model', 1) > 1 else whisper_config_dict.get('num_workers', 1)
        
        model = WhisperModel(
            model_id,
            device=whisper_config_dict.get('inference_device', 'auto'),
            device_index=whisper_config_dict.get('device_index', 0),
            compute_type=whisper_config_dict.get('compute_type', 'default'),
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )
        
        # Perform transcription
        segments, transcription_info = model.transcribe(audio_data, **kwargs)
        segments = TranscriptionSegment.from_faster_whisper_segments(segments)
        
        end_time = time.perf_counter()
        logger.info(f"Process worker {worker_id}: transcribed audio in {end_time - start_time:.2f} seconds using model {model_id}")
        
        return segments, transcription_info
        
    except Exception as e:
        logger.error(f"Process worker {worker_id}: transcription failed: {e}")
        raise


class CPUAffinityThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that assigns specific CPU cores to each worker thread."""
    
    def __init__(self, max_workers=None, enable_cpu_affinity=True, *args, **kwargs):
        self.enable_cpu_affinity = enable_cpu_affinity
        if self.enable_cpu_affinity:
            self.cpu_count = os.cpu_count() or 4
            if max_workers:
                self.cores_per_worker = max(1, self.cpu_count // max_workers)
            else:
                self.cores_per_worker = 1
            logger.info(f"CPU affinity ThreadPool: {self.cpu_count} cores, {self.cores_per_worker} cores per worker")
        
        super().__init__(max_workers=max_workers, *args, **kwargs)
        self.worker_core_map = {}
    
    def submit(self, fn, *args, **kwargs):
        if self.enable_cpu_affinity:
            # Wrap the function to set CPU affinity
            def wrapped_fn(*args, **kwargs):
                thread_id = threading.get_ident()
                
                # Assign cores to this worker thread if not done already
                if thread_id not in self.worker_core_map:
                    worker_index = len(self.worker_core_map)
                    start_core = worker_index * self.cores_per_worker
                    end_core = min(start_core + self.cores_per_worker, self.cpu_count)
                    assigned_cores = list(range(start_core, end_core))
                    self.worker_core_map[thread_id] = assigned_cores
                    
                    try:
                        os.sched_setaffinity(0, assigned_cores)
                        logger.info(f"Worker thread {thread_id} bound to CPU cores: {assigned_cores}")
                    except (AttributeError, OSError) as e:
                        logger.warning(f"Could not set CPU affinity for worker {thread_id}: {e}")
                
                return fn(*args, **kwargs)
            
            return super().submit(wrapped_fn, *args, **kwargs)
        else:
            return super().submit(fn, *args, **kwargs)


# TODO: enable concurrent model downloads


class ModelPool:
    """Pool of identical model instances for parallel processing."""
    
    def __init__(self, model_id: str, load_fn, pool_size: int, ttl: int, unload_callback):
        self.model_id = model_id
        self.load_fn = load_fn
        self.pool_size = pool_size
        self.ttl = ttl
        self.unload_callback = unload_callback
        self.instances: list[SelfDisposingModel[WhisperModel]] = []
        self.available = queue.Queue(maxsize=pool_size)  # Use threading.Queue, not asyncio.Queue
        self._lock = threading.Lock()
        self._initialized = False
    
    def _initialize_pool(self):
        """Initialize the pool with model instances."""
        with self._lock:
            if self._initialized:
                return
            
            for i in range(self.pool_size):
                instance_id = f"{self.model_id}#{i}"
                model_instance = SelfDisposingModel[WhisperModel](
                    instance_id,
                    load_fn=self.load_fn,
                    ttl=self.ttl,
                    model_unloaded_callback=lambda mid=instance_id: self._handle_instance_unloaded(mid),
                )
                self.instances.append(model_instance)
                # Pre-populate the queue with all instances
                self.available.put(model_instance)
            
            self._initialized = True
            logger.info(f"Initialized model pool for {self.model_id} with {self.pool_size} instances")
    
    def _handle_instance_unloaded(self, instance_id: str):
        """Handle when a model instance is unloaded."""
        logger.debug(f"Model instance {instance_id} unloaded")
        # If all instances are unloaded, notify the parent manager
        if all(instance.model is None for instance in self.instances):
            self.unload_callback(self.model_id)
    
    @contextlib.contextmanager
    def get_model(self):
        """Get an available model instance from the pool."""
        if not self._initialized:
            self._initialize_pool()
        
        # Get an available model instance (blocking)
        model_instance = self.available.get()
        try:
            yield model_instance
        finally:
            # Return the instance to the pool
            self.available.put(model_instance)
    
    def unload_all(self):
        """Unload all model instances in the pool."""
        with self._lock:
            for instance in self.instances:
                try:
                    instance.unload()
                except Exception as e:
                    logger.error(f"Error unloading model instance: {e}")
            self.instances.clear()
            # Clear the queue
            while not self.available.empty():
                try:
                    self.available.get_nowait()
                except queue.Empty:
                    break


class WhisperModelManager:
    def __init__(self, whisper_config: WhisperConfig) -> None:
        self.whisper_config = whisper_config
        self.loaded_models: OrderedDict[str, SelfDisposingModel[WhisperModel]] = OrderedDict()
        self.model_pools: OrderedDict[str, ModelPool] = OrderedDict()
        
        # Choose executor type based on configuration
        if whisper_config.use_process_pool:
            self.executor = ProcessPoolExecutor(max_workers=whisper_config.max_concurrent_jobs)
            self._use_process_pool = True
            logger.info(f"Using ProcessPoolExecutor with {whisper_config.max_concurrent_jobs} workers for GIL bypass")
        else:
            self.executor = CPUAffinityThreadPoolExecutor(
                max_workers=whisper_config.max_concurrent_jobs,
                enable_cpu_affinity=whisper_config.enable_cpu_affinity
            )
            self._use_process_pool = False
            logger.info(f"Using ThreadPoolExecutor with {whisper_config.max_concurrent_jobs} workers")
        
        self._lock = threading.Lock()
        self._worker_counter = 0

    def _load_fn(self, model_id: str) -> WhisperModel:
        # Use single-threaded mode per instance to avoid threading conflicts
        cpu_threads = 1 if self.whisper_config.model_instances_per_model > 1 else self.whisper_config.cpu_threads
        num_workers = 1 if self.whisper_config.model_instances_per_model > 1 else self.whisper_config.num_workers
        
        return WhisperModel(
            model_id,
            device=self.whisper_config.inference_device,
            device_index=self.whisper_config.device_index,
            compute_type=self.whisper_config.compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )

    def _handle_model_unloaded(self, model_id: str) -> None:
        with self._lock:
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]

    def _handle_pool_unloaded(self, model_id: str) -> None:
        with self._lock:
            if model_id in self.model_pools:
                del self.model_pools[model_id]

    def unload_model(self, model_id: str) -> None:
        with self._lock:
            # Check if it's a pooled model
            if model_id in self.model_pools:
                self.model_pools[model_id].unload_all()
                del self.model_pools[model_id]
            # Check if it's a single model
            elif model_id in self.loaded_models:
                model = self.loaded_models.get(model_id)
                if model is None:
                    raise KeyError(f"Model {model_id} not found")
                # WARN: ~300 MB of memory will still be held by the model. See https://github.com/SYSTRAN/faster-whisper/issues/992
                self.loaded_models[model_id].unload()
            else:
                raise KeyError(f"Model {model_id} not found")

    def load_model(self, model_id: str) -> SelfDisposingModel[WhisperModel]:
        """Load a single model instance (legacy compatibility)."""
        logger.debug(f"Loading model {model_id}")
        with self._lock:
            logger.debug("Acquired lock")
            if model_id in self.loaded_models:
                logger.debug(f"{model_id} model already loaded")
                return self.loaded_models[model_id]
            self.loaded_models[model_id] = SelfDisposingModel[WhisperModel](
                model_id,
                load_fn=lambda: self._load_fn(model_id),
                ttl=self.whisper_config.ttl,
                model_unloaded_callback=self._handle_model_unloaded,
            )
            return self.loaded_models[model_id]

    def get_model_from_pool(self, model_id: str):
        """Get a model instance from the pool for parallel processing."""
        with self._lock:
            if model_id not in self.model_pools:
                # Create a new pool for this model
                self.model_pools[model_id] = ModelPool(
                    model_id=model_id,
                    load_fn=lambda: self._load_fn(model_id),
                    pool_size=self.whisper_config.model_instances_per_model,
                    ttl=self.whisper_config.ttl,
                    unload_callback=self._handle_pool_unloaded,
                )
        
        return self.model_pools[model_id].get_model()

    async def transcribe_async(self, model_id: str, audio, **kwargs):
        """Perform asynchronous transcription using the thread pool."""
        def _transcribe():
            # Use the pool for parallel processing
            async def _async_transcribe():
                async with self.get_model_from_pool(model_id) as model_instance:
                    with model_instance as whisper:
                        return whisper.transcribe(audio, **kwargs)
            
            # Run the async function in the current thread's event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(_async_transcribe())
            finally:
                loop.close()
        
        # Execute in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _transcribe)
