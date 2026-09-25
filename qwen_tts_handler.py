"""
Qwen3-TTS Handler for Ultimate TTS Studio
Provides integration with Qwen3-TTS Text-to-Speech system
Supports three modes: Voice Design, Voice Clone, and Custom Voice
"""

import os
import sys
import warnings
import numpy as np
import torch
import gc
import re
import random
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Import for audio file saving
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    try:
        from scipy.io.wavfile import write as wav_write
        SOUNDFILE_AVAILABLE = False
    except ImportError:
        SOUNDFILE_AVAILABLE = None

# Import for MP3 conversion
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Add qwen_tts to path
current_dir = os.path.dirname(os.path.abspath(__file__))
qwen_tts_path = os.path.join(current_dir, 'qwen_tts')
if qwen_tts_path not in sys.path:
    sys.path.insert(0, qwen_tts_path)

# Try to import Qwen TTS
try:
    from qwen_tts import Qwen3TTSModel
    QWEN_TTS_AVAILABLE = True
    print("✅ Qwen TTS handler loaded")
except ImportError as e:
    QWEN_TTS_AVAILABLE = False
    # Check if it's a transformers compatibility issue
    if "auto_docstring" in str(e) or "transformers" in str(e).lower():
        print(f"⚠️ Qwen3-TTS not available: transformers version incompatibility - {e}")
    else:
        print(f"⚠️ Qwen3-TTS not available: {e}")

# Try to import whisper for transcription
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# Model configurations
QWEN_TTS_MODELS = {
    "VoiceDesign": {
        "sizes": ["1.7B"],
        "description": "Create custom voices using natural language descriptions",
        "supports_chunking": False,
        "supports_conversation": False,
        "supports_ebook": False
    },
    "Base": {
        "sizes": ["0.6B", "1.7B"],
        "description": "Voice cloning from reference audio",
        "supports_chunking": True,
        "supports_conversation": True,
        "supports_ebook": True
    },
    "CustomVoice": {
        "sizes": ["0.6B", "1.7B"],
        "description": "TTS with predefined speakers and style instructions",
        "supports_chunking": False,
        "supports_conversation": False,
        "supports_ebook": False
    }
}

# Speaker and language choices for CustomVoice model
QWEN_SPEAKERS = ["Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"]
QWEN_LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]


def get_model_repo_id(model_type: str, model_size: str) -> str:
    """Get HuggingFace repo ID for a model."""
    return f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}"


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def chunk_text(text: str, max_chars: int = 200) -> List[str]:
    """
    Split text into chunks without cutting words.
    Tries to split on sentence boundaries first, then falls back to word boundaries.
    """
    text = text.strip()
    if not text:
        return []
    
    if len(text) <= max_chars:
        return [text]
    
    # Sentence-ending punctuation patterns
    sentence_endings = re.compile(r'(?<=[.!?。！？])\s+')
    
    # Split into sentences first
    sentences = sentence_endings.split(text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # If single sentence is too long, split by words
        if len(sentence) > max_chars:
            # Flush current chunk first
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split long sentence by words
            words = sentence.split()
            for word in words:
                if len(current_chunk) + len(word) + 1 <= max_chars:
                    current_chunk = current_chunk + " " + word if current_chunk else word
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = word
        else:
            # Try to add sentence to current chunk
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            if len(test_chunk) <= max_chars:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


class QwenTTSHandler:
    """Handler for Qwen3-TTS system"""
    
    def __init__(self):
        self.loaded_models: Dict[Tuple[str, str], Any] = {}
        self.current_model_key: Optional[Tuple[str, str]] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = None
        
        # Set up checkpoints directory structure
        self.checkpoints_dir = Path('checkpoints') / 'qwen_tts'
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.checkpoints_dir / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_device(self) -> str:
        """Get the current device"""
        return self.device
    
    def _get_model_local_path(self, model_type: str, model_size: str) -> Path:
        """Get the local path for a specific model."""
        # Create a clean folder name from the model type and size
        folder_name = f"{model_type}_{model_size}".replace(".", "_")
        return self.checkpoints_dir / folder_name
    
    def check_model_downloaded(self, model_type: str, model_size: str) -> bool:
        """Check if a model is completely downloaded (including weights) in the checkpoints folder."""
        local_path = self._get_model_local_path(model_type, model_size)
        if not (local_path / "config.json").exists():
            return False
        has_weights = (
            (local_path / "model.safetensors").exists()
            or (local_path / "pytorch_model.bin").exists()
            or (local_path / "model.safetensors.index.json").exists()
        )
        has_tokenizer_weights = (
            not (local_path / "speech_tokenizer").exists()
            or (local_path / "speech_tokenizer" / "model.safetensors").exists()
            or (local_path / "speech_tokenizer" / "pytorch_model.bin").exists()
        )
        return has_weights and has_tokenizer_weights
    
    def get_downloaded_models_status(self) -> str:
        """Get status of all available models."""
        lines = ["### Qwen3-TTS Model Download Status\n"]
        for model_type, info in QWEN_TTS_MODELS.items():
            lines.append(f"**{model_type}** - {info['description']}")
            for size in info["sizes"]:
                status = "✅ Downloaded" if self.check_model_downloaded(model_type, size) else "⬜ Not downloaded"
                local_path = self._get_model_local_path(model_type, size)
                lines.append(f"  - {size}: {status}")
            lines.append("")
        return "\n".join(lines)
    
    def download_model(self, model_type: str, model_size: str, progress_callback=None) -> Tuple[bool, str]:
        """Download a specific model to the checkpoints folder."""
        if model_size not in QWEN_TTS_MODELS.get(model_type, {}).get("sizes", []):
            return False, f"❌ Invalid combination: {model_type} {model_size}"
        
        repo_id = get_model_repo_id(model_type, model_size)
        local_path = self._get_model_local_path(model_type, model_size)
        
        if self.check_model_downloaded(model_type, model_size):
            return True, f"✅ {model_type} {model_size} is already downloaded at {local_path}!"
        
        try:
            from huggingface_hub import snapshot_download
            
            if progress_callback:
                progress_callback(0, desc=f"Downloading {model_type} {model_size}...")
            
            print(f"📥 Downloading {repo_id} to {local_path}...")
            
            # Download to the local checkpoints folder
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_path),
                local_dir_use_symlinks=False,
                cache_dir=str(self.cache_dir),
            )
            
            if progress_callback:
                progress_callback(1, desc="Complete!")
            
            return True, f"✅ Successfully downloaded {model_type} {model_size} to {local_path}!"
        except Exception as e:
            return False, f"❌ Error downloading {model_type} {model_size}: {str(e)}"
    
    def load_model(self, model_type: str, model_size: str) -> Tuple[bool, str]:
        """Load a specific model into memory from the checkpoints folder."""
        if not QWEN_TTS_AVAILABLE:
            return False, "❌ Qwen3-TTS not available"
        
        if model_size not in QWEN_TTS_MODELS.get(model_type, {}).get("sizes", []):
            return False, f"❌ Invalid combination: {model_type} {model_size}"
        
        key = (model_type, model_size)
        
        if key in self.loaded_models:
            self.current_model_key = key
            return True, f"✅ {model_type} {model_size} is already loaded!"
        
        try:
            local_path = self._get_model_local_path(model_type, model_size)
            
            # Check if model is downloaded locally
            if not self.check_model_downloaded(model_type, model_size):
                print(f"📥 Model not found locally, downloading {model_type} {model_size}...")
                success, msg = self.download_model(model_type, model_size)
                if not success:
                    return False, msg
            
            print(f"🔄 Loading Qwen3-TTS {model_type} {model_size} from {local_path}...")
            
            model = Qwen3TTSModel.from_pretrained(
                str(local_path),
                device_map=self.device,
                torch_dtype=torch.bfloat16,
            )
            
            self.loaded_models[key] = model
            self.current_model_key = key
            
            print(f"✅ Qwen3-TTS {model_type} {model_size} loaded successfully")
            return True, f"✅ Successfully loaded {model_type} {model_size}!"
        except Exception as e:
            print(f"❌ Error loading Qwen3-TTS: {e}")
            import traceback
            traceback.print_exc()
            return False, f"❌ Error loading {model_type} {model_size}: {str(e)}"
    
    def unload_model(self, model_type: str = None, model_size: str = None) -> str:
        """Unload a specific model or all models from memory."""
        if model_type is None or model_size is None:
            # Unload all models
            if not self.loaded_models:
                return "⚠️ No models are currently loaded."
            
            count = len(self.loaded_models)
            self.loaded_models.clear()
            self.current_model_key = None
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return f"✅ Unloaded {count} Qwen3-TTS model(s) and freed GPU memory."
        else:
            key = (model_type, model_size)
            if key not in self.loaded_models:
                return f"⚠️ {model_type} {model_size} is not loaded."
            
            del self.loaded_models[key]
            if self.current_model_key == key:
                self.current_model_key = None
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return f"✅ Unloaded {model_type} {model_size} and freed GPU memory."
    
    def get_loaded_models_status(self) -> str:
        """Get status of currently loaded models in memory."""
        if not self.loaded_models:
            return "No Qwen3-TTS models currently loaded in memory."
        
        lines = ["**Currently loaded Qwen3-TTS models:**"]
        for (model_type, model_size) in self.loaded_models.keys():
            current = " (active)" if (model_type, model_size) == self.current_model_key else ""
            lines.append(f"- {model_type} ({model_size}){current}")
        return "\n".join(lines)
    
    def get_model(self, model_type: str, model_size: str) -> Any:
        """Get or load a model by type and size."""
        key = (model_type, model_size)
        
        if key not in self.loaded_models:
            success, message = self.load_model(model_type, model_size)
            if not success:
                raise RuntimeError(message)
        
        self.current_model_key = key
        return self.loaded_models[key]
    
    def get_whisper_model(self):
        """Load Whisper tiny model for transcription."""
        if not WHISPER_AVAILABLE:
            return None
        
        if self.whisper_model is None:
            # Store whisper model in checkpoints folder
            whisper_cache_dir = self.checkpoints_dir / "whisper"
            whisper_cache_dir.mkdir(parents=True, exist_ok=True)
            self.whisper_model = whisper.load_model(
                "tiny", 
                device=self.device,
                download_root=str(whisper_cache_dir)
            )
        return self.whisper_model
    
    def unload_whisper(self):
        """Force unload whisper model from GPU."""
        if self.whisper_model is not None:
            try:
                self.whisper_model.cpu()
            except:
                pass
            self.whisper_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    def transcribe_audio(self, audio) -> str:
        """Transcribe audio using Whisper tiny."""
        if not WHISPER_AVAILABLE:
            return "❌ Whisper not available for transcription"
        
        if audio is None:
            return "Please upload audio first."
        
        try:
            import librosa
            
            # Handle different audio input formats from Gradio
            if isinstance(audio, str):
                # File path
                wav, sr = librosa.load(audio, sr=None, mono=True)
            elif isinstance(audio, tuple):
                if len(audio) == 2:
                    # Could be (sr, wav) or (wav, sr) depending on Gradio version
                    if isinstance(audio[0], int) or (isinstance(audio[0], (float, np.floating)) and audio[0] > 1000):
                        # First element is sample rate
                        sr, wav = audio
                    else:
                        # First element is waveform
                        wav, sr = audio
                else:
                    return f"Transcription error: unexpected audio tuple length {len(audio)}"
                wav = np.array(wav).astype(np.float32)
            elif isinstance(audio, dict):
                # Newer Gradio format: {"path": ..., "array": ..., "sampling_rate": ...}
                if "path" in audio and audio["path"]:
                    wav, sr = librosa.load(audio["path"], sr=None, mono=True)
                elif "array" in audio:
                    wav = np.array(audio["array"]).astype(np.float32)
                    sr = audio.get("sampling_rate", 16000)
                else:
                    return "Transcription error: invalid audio dict format"
            else:
                return f"Transcription error: unsupported audio type {type(audio)}"
            
            wav = wav.astype(np.float32)
            
            # Normalize
            max_val = np.abs(wav).max()
            if max_val > 1.0:
                wav = wav / max_val
            
            # Convert to mono if needed
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)
            
            # Resample to 16kHz for Whisper
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            
            model = self.get_whisper_model()
            result = model.transcribe(wav, fp16=torch.cuda.is_available())
            text = result["text"].strip()
            
            # Unload whisper to free GPU memory
            self.unload_whisper()
            
            return text
        except Exception as e:
            self.unload_whisper()
            import traceback
            traceback.print_exc()
            return f"Transcription error: {str(e)}"
    
    def _normalize_audio(self, wav, eps=1e-12, clip=True):
        """Normalize audio to float32 in [-1, 1] range."""
        x = np.asarray(wav)
        
        if np.issubdtype(x.dtype, np.integer):
            info = np.iinfo(x.dtype)
            if info.min < 0:
                y = x.astype(np.float32) / max(abs(info.min), info.max)
            else:
                mid = (info.max + 1) / 2.0
                y = (x.astype(np.float32) - mid) / mid
        elif np.issubdtype(x.dtype, np.floating):
            y = x.astype(np.float32)
            m = np.max(np.abs(y)) if y.size else 0.0
            if m > 1.0 + 1e-6:
                y = y / (m + eps)
        else:
            raise TypeError(f"Unsupported dtype: {x.dtype}")
        
        if clip:
            y = np.clip(y, -1.0, 1.0)
        
        if y.ndim > 1:
            y = np.mean(y, axis=-1).astype(np.float32)
        
        return y
    
    def _audio_to_tuple(self, audio):
        """Convert Gradio audio input to (wav, sr) tuple."""
        if audio is None:
            return None
        
        if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
            sr, wav = audio
            wav = self._normalize_audio(wav)
            return wav, int(sr)
        
        if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
            sr = int(audio["sampling_rate"])
            wav = self._normalize_audio(audio["data"])
            return wav, sr
        
        # If it's a file path
        if isinstance(audio, str) and os.path.exists(audio):
            import librosa
            wav, sr = librosa.load(audio, sr=None, mono=True)
            return wav.astype(np.float32), int(sr)
        
        return None

    
    def generate_voice_design(
        self,
        text: str,
        language: str = "Auto",
        voice_description: str = "",
        seed: int = -1,
        max_new_tokens: int = 2048
    ) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
        """
        Generate speech using Voice Design model (1.7B only).
        Creates unique voices from natural language descriptions.
        Does NOT support chunking - for short texts only.
        """
        if not QWEN_TTS_AVAILABLE:
            return None, "❌ Qwen3-TTS not available"
        
        if not text or not text.strip():
            return None, "❌ Error: Text is required."
        
        if not voice_description or not voice_description.strip():
            return None, "❌ Error: Voice description is required."
        
        try:
            # Handle seed
            if seed == -1:
                seed = random.randint(0, 2147483647)
            seed = int(seed)
            set_seed(seed)
            
            tts = self.get_model("VoiceDesign", "1.7B")
            
            print(f"\n{'='*50}")
            print(f"🎨 Qwen Voice Design Generation")
            print(f"{'='*50}")
            print(f"🎲 Seed: {seed}")
            print(f"📝 Text length: {len(text)} chars")
            print(f"🎭 Voice description: {voice_description[:50]}...")
            
            wavs, sr = tts.generate_voice_design(
                text=text.strip(),
                language=language,
                instruct=voice_description.strip(),
                non_streaming_mode=True,
                max_new_tokens=max_new_tokens,
            )
            
            total_duration = len(wavs[0]) / sr
            print(f"\n{'='*50}")
            print(f"✅ Complete! Duration: {total_duration:.2f}s")
            print(f"{'='*50}\n")
            
            status = f"Generated {total_duration:.1f}s of audio | Seed: {seed}"
            return (sr, wavs[0]), status
            
        except Exception as e:
            print(f"❌ Error: {type(e).__name__}: {e}")
            return None, f"❌ Error: {type(e).__name__}: {e}"
    
    def generate_voice_clone(
        self,
        text: str,
        ref_audio,
        ref_text: str = "",
        language: str = "Auto",
        use_xvector_only: bool = False,
        model_size: str = "1.7B",
        max_chunk_chars: int = 200,
        chunk_gap: float = 0.0,
        seed: int = -1,
        max_new_tokens: int = 2048
    ) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
        """
        Generate speech using Base (Voice Clone) model.
        Supports chunking for long texts.
        Can be used in conversation mode and ebook mode.
        """
        if not QWEN_TTS_AVAILABLE:
            return None, "❌ Qwen3-TTS not available"
        
        if not text or not text.strip():
            return None, "❌ Error: Target text is required."
        
        audio_tuple = self._audio_to_tuple(ref_audio)
        if audio_tuple is None:
            return None, "❌ Error: Reference audio is required."
        
        if not use_xvector_only and (not ref_text or not ref_text.strip()):
            return None, "❌ Error: Reference text is required when 'Use x-vector only' is not enabled."
        
        try:
            from tqdm import tqdm
            
            # Handle seed
            if seed == -1:
                seed = random.randint(0, 2147483647)
            seed = int(seed)
            
            tts = self.get_model("Base", model_size)
            chunks = chunk_text(text.strip(), max_chars=int(max_chunk_chars))
            
            print(f"\n{'='*50}")
            print(f"🎭 Qwen Voice Clone Generation ({model_size})")
            print(f"{'='*50}")
            print(f"🎲 Seed: {seed}")
            print(f"📝 Text length: {len(text)} chars → {len(chunks)} chunk(s)")
            print(f"⏱️ Chunk gap: {chunk_gap}s")
            
            all_wavs = []
            sr = None
            
            for i, chunk in enumerate(tqdm(chunks, desc="Generating chunks", unit="chunk")):
                # Set seed before each chunk for consistency
                set_seed(seed)
                
                print(f"\n🔊 Chunk {i+1}/{len(chunks)} [Seed: {seed}]: \"{chunk[:50]}{'...' if len(chunk) > 50 else ''}\"")
                
                wavs, sr = tts.generate_voice_clone(
                    text=chunk,
                    language=language,
                    ref_audio=audio_tuple,
                    ref_text=ref_text.strip() if ref_text else None,
                    x_vector_only_mode=use_xvector_only,
                    max_new_tokens=max_new_tokens,
                )
                
                all_wavs.append(wavs[0])
                print(f"   ✅ Generated {len(wavs[0])/sr:.2f}s of audio")
            
            # Concatenate all audio chunks with gap (silence) between them
            if len(all_wavs) > 1 and chunk_gap > 0:
                gap_samples = int(sr * chunk_gap)
                silence = np.zeros(gap_samples, dtype=np.float32)
                chunks_with_gaps = []
                for i, wav in enumerate(all_wavs):
                    chunks_with_gaps.append(wav)
                    if i < len(all_wavs) - 1:  # Don't add gap after last chunk
                        chunks_with_gaps.append(silence)
                final_wav = np.concatenate(chunks_with_gaps)
            else:
                final_wav = np.concatenate(all_wavs) if len(all_wavs) > 1 else all_wavs[0]
            
            total_duration = len(final_wav) / sr
            print(f"\n{'='*50}")
            print(f"✅ Complete! Total duration: {total_duration:.2f}s")
            print(f"{'='*50}\n")
            
            status = f"Generated {len(chunks)} chunk(s), {total_duration:.1f}s total | Seed: {seed}" if len(chunks) > 1 else f"Generated {total_duration:.1f}s of audio | Seed: {seed}"
            return (sr, final_wav), status
            
        except Exception as e:
            print(f"❌ Error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None, f"❌ Error: {type(e).__name__}: {e}"
    
    def generate_custom_voice(
        self,
        text: str,
        speaker: str = "Ryan",
        language: str = "Auto",
        instruct: str = "",
        model_size: str = "1.7B",
        seed: int = -1,
        max_new_tokens: int = 2048
    ) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
        """
        Generate speech using CustomVoice model.
        Uses predefined speakers with optional style instructions.
        Does NOT support chunking - for short texts only.
        """
        if not QWEN_TTS_AVAILABLE:
            return None, "❌ Qwen3-TTS not available"
        
        if not text or not text.strip():
            return None, "❌ Error: Text is required."
        
        if not speaker:
            return None, "❌ Error: Speaker is required."
        
        try:
            # Handle seed
            if seed == -1:
                seed = random.randint(0, 2147483647)
            seed = int(seed)
            set_seed(seed)
            
            tts = self.get_model("CustomVoice", model_size)
            
            print(f"\n{'='*50}")
            print(f"🗣️ Qwen Custom Voice Generation ({model_size})")
            print(f"{'='*50}")
            print(f"🎲 Seed: {seed}")
            print(f"👤 Speaker: {speaker}")
            chunks = split_text_into_chunks(text.strip(), max_chars=250)
            print(f"📝 Text length: {len(text)} chars -> {len(chunks)} chunk(s)")
            
            all_wavs = []
            sr = 24000
            for idx, chunk_text in enumerate(chunks):
                set_seed(seed + idx)
                print(f"  [{idx+1}/{len(chunks)}] Generating ({len(chunk_text)} chars)...")
                wavs, sr = tts.generate_custom_voice(
                    text=chunk_text,
                    language=language,
                    speaker=speaker.lower().replace(" ", "_"),
                    instruct=instruct.strip() if instruct else None,
                    non_streaming_mode=True,
                    max_new_tokens=max_new_tokens,
                )
                all_wavs.append(wavs[0])
                if idx < len(chunks) - 1:
                    all_wavs.append(np.zeros(int(sr * 0.25), dtype=np.float32))
            
            final_wav = np.concatenate(all_wavs) if len(all_wavs) > 1 else all_wavs[0]
            total_duration = len(final_wav) / sr
            print(f"\n{'='*50}")
            print(f"✅ Complete! Duration: {total_duration:.2f}s ({len(chunks)} chunks)")
            print(f"{'='*50}\n")
            
            status = f"Generated {len(chunks)} chunk(s), {total_duration:.1f}s total | Seed: {seed}"
            return (sr, final_wav), status
            
        except Exception as e:
            print(f"❌ Error: {type(e).__name__}: {e}")
            return None, f"❌ Error: {type(e).__name__}: {e}"


# Global handler instance
_qwen_tts_handler: Optional[QwenTTSHandler] = None


def get_qwen_tts_handler() -> QwenTTSHandler:
    """Get or create the global Qwen TTS handler instance."""
    global _qwen_tts_handler
    if _qwen_tts_handler is None:
        _qwen_tts_handler = QwenTTSHandler()
    return _qwen_tts_handler


def init_qwen_tts(model_type: str = "Base", model_size: str = "1.7B") -> Tuple[bool, str]:
    """Initialize Qwen TTS model."""
    if not QWEN_TTS_AVAILABLE:
        return False, "❌ Qwen3-TTS not available"
    
    handler = get_qwen_tts_handler()
    return handler.load_model(model_type, model_size)


def unload_qwen_tts(model_type: str = None, model_size: str = None) -> str:
    """Unload Qwen TTS model(s)."""
    handler = get_qwen_tts_handler()
    return handler.unload_model(model_type, model_size)


def get_qwen_tts_status() -> str:
    """Get Qwen TTS status."""
    if not QWEN_TTS_AVAILABLE:
        return "❌ Qwen3-TTS not available"
    
    handler = get_qwen_tts_handler()
    return handler.get_loaded_models_status()


def transcribe_qwen_audio(audio) -> str:
    """Transcribe audio using Whisper."""
    handler = get_qwen_tts_handler()
    return handler.transcribe_audio(audio)


# Main generation functions for integration with launch.py

def generate_qwen_voice_design_tts(
    text: str,
    language: str = "Auto",
    voice_description: str = "",
    seed: int = -1,
    effects_settings: Dict = None,
    audio_format: str = "wav",
    skip_file_saving: bool = False
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    """Generate TTS using Qwen Voice Design mode."""
    handler = get_qwen_tts_handler()
    result, status = handler.generate_voice_design(
        text=text,
        language=language,
        voice_description=voice_description,
        seed=seed
    )
    
    if result is None:
        return None, status
    
    # Apply effects if provided (placeholder for future implementation)
    # if effects_settings:
    #     result = apply_audio_effects(result, effects_settings)
    
    if not skip_file_saving:
        # Save to file
        sr, audio_data = result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"qwen_voice_design_{timestamp}"
        # File saving would be handled by the main app
    
    return result, status


def generate_qwen_voice_clone_tts(
    text: str,
    ref_audio,
    ref_text: str = "",
    language: str = "Auto",
    use_xvector_only: bool = False,
    model_size: str = "1.7B",
    max_chunk_chars: int = 200,
    chunk_gap: float = 0.0,
    seed: int = -1,
    effects_settings: Dict = None,
    audio_format: str = "wav",
    skip_file_saving: bool = False
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    """Generate TTS using Qwen Voice Clone mode."""
    handler = get_qwen_tts_handler()
    result, status = handler.generate_voice_clone(
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        language=language,
        use_xvector_only=use_xvector_only,
        model_size=model_size,
        max_chunk_chars=max_chunk_chars,
        chunk_gap=chunk_gap,
        seed=seed
    )
    
    if result is None:
        return None, status
    
    return result, status


def generate_qwen_custom_voice_tts(
    text: str,
    speaker: str = "Ryan",
    language: str = "Auto",
    instruct: str = "",
    model_size: str = "1.7B",
    seed: int = -1,
    effects_settings: Dict = None,
    audio_format: str = "wav",
    skip_file_saving: bool = False
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    """Generate TTS using Qwen Custom Voice mode."""
    handler = get_qwen_tts_handler()
    result, status = handler.generate_custom_voice(
        text=text,
        speaker=speaker,
        language=language,
        instruct=instruct,
        model_size=model_size,
        seed=seed
    )
    
    if result is None:
        return None, status
    
    return result, status
