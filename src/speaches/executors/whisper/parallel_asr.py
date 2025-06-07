from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from faster_whisper.transcribe import BatchedInferencePipeline

from speaches.api_types import TranscriptionSegment

if TYPE_CHECKING:
    from collections.abc import Iterable
    from faster_whisper.transcribe import TranscriptionInfo
    from speaches.executors.whisper.model_manager import WhisperModelManager

logger = logging.getLogger(__name__)


class ParallelWhisperASR:
    """ASR executor that supports parallel transcription jobs."""
    
    def __init__(self, model_manager: WhisperModelManager, use_batched_mode: bool = False):
        self.model_manager = model_manager
        self.use_batched_mode = use_batched_mode
    
    async def transcribe_async(
        self,
        model_id: str,
        audio,
        **kwargs
    ) -> tuple[Iterable[TranscriptionSegment], TranscriptionInfo]:
        """Perform async transcription using either process or thread pool."""
        start = time.perf_counter()
        
        # Choose execution method based on model manager configuration
        if self.model_manager._use_process_pool:
            # Process-based execution (bypasses GIL)
            worker_id = self.model_manager._worker_counter
            self.model_manager._worker_counter = (self.model_manager._worker_counter + 1) % self.model_manager.whisper_config.max_concurrent_jobs
            
            # Convert config to dict for process serialization
            config_dict = {
                'inference_device': self.model_manager.whisper_config.inference_device,
                'device_index': self.model_manager.whisper_config.device_index,
                'compute_type': self.model_manager.whisper_config.compute_type,
                'cpu_threads': self.model_manager.whisper_config.cpu_threads,
                'num_workers': self.model_manager.whisper_config.num_workers,
                'model_instances_per_model': self.model_manager.whisper_config.model_instances_per_model,
                'max_concurrent_jobs': self.model_manager.whisper_config.max_concurrent_jobs,
                'enable_cpu_affinity': self.model_manager.whisper_config.enable_cpu_affinity,
            }
            
            # Import the process function here to avoid circular imports
            from speaches.executors.whisper.model_manager import _process_transcribe
            
            loop = asyncio.get_event_loop()
            
            # Create a partial function with all the arguments
            import functools
            process_func = functools.partial(
                _process_transcribe,
                model_id,
                audio,
                worker_id,
                config_dict,
                **kwargs
            )
            
            segments, transcription_info = await loop.run_in_executor(
                self.model_manager.executor,
                process_func
            )
        else:
            # Thread-based execution (original implementation)
            def _transcribe():
                with self.model_manager.get_model_from_pool(model_id) as model_instance:
                    with model_instance as whisper:
                        whisper_model = BatchedInferencePipeline(model=whisper) if self.use_batched_mode else whisper
                        segments, transcription_info = whisper_model.transcribe(audio, **kwargs)
                        segments = TranscriptionSegment.from_faster_whisper_segments(segments)
                        return segments, transcription_info
            
            # Execute in thread pool
            loop = asyncio.get_event_loop()
            segments, transcription_info = await loop.run_in_executor(
                self.model_manager.executor, 
                _transcribe
            )
        
        end = time.perf_counter()
        logger.info(f"Transcribed audio in {end - start:.2f} seconds using model {model_id}")
        
        return segments, transcription_info
    
    def transcribe_sync(
        self,
        model_id: str,
        audio,
        **kwargs
    ) -> tuple[Iterable[TranscriptionSegment], TranscriptionInfo]:
        """Synchronous transcription (legacy compatibility)."""
        with self.model_manager.load_model(model_id) as whisper:
            whisper_model = BatchedInferencePipeline(model=whisper) if self.use_batched_mode else whisper
            segments, transcription_info = whisper_model.transcribe(audio, **kwargs)
            segments = TranscriptionSegment.from_faster_whisper_segments(segments)
            return segments, transcription_info