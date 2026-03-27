"""
Fish Speech S2 Pro Handler for Ultimate TTS Studio
Provides integration with Fish Audio S2 Pro (4B parameter) TTS model.

S2 Pro uses a Dual-AR architecture with fish_qwen3_omni model type,
which requires the latest fish-speech codebase (separate from the S1 code).

This handler manages:
- Cloning/updating the latest fish-speech repo into fish_speech_s2/
- Downloading S2 Pro model weights
- Loading/unloading the model
- TTS inference with reference audio support and inline emotion control
"""

import os
import sys
import gc
import queue
import warnings
import subprocess
import tempfile
import threading
import time
import traceback
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime

warnings.filterwarnings('ignore')

# ===== Configuration =====
S2_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fish_speech_s2")
S2_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "s2-pro")
S2_REPO_URL = "https://github.com/fishaudio/fish-speech.git"
S2_HF_REPO = "fishaudio/s2-pro"

# Global state
FISH_S2_AVAILABLE = False
FISH_S2_ENGINE = None
FISH_S2_LLAMA_QUEUE = None
FISH_S2_LOADED = False
FISH_S2_LOADING = False
_s2_modules_imported = False
# Snapshot of sys.path BEFORE any S2 imports, used to undo pyrootutils pollution
_pre_s2_sys_path = None


def _add_s2_to_path():
    """Add the S2 fish-speech repo to sys.path (temporarily, at front)."""
    if S2_REPO_DIR not in sys.path:
        sys.path.insert(0, S2_REPO_DIR)


def _remove_s2_from_path():
    """Remove the S2 fish-speech repo from sys.path."""
    while S2_REPO_DIR in sys.path:
        sys.path.remove(S2_REPO_DIR)
    # Also remove the resolved absolute path variant (pyrootutils may add it)
    resolved = str(Path(S2_REPO_DIR).resolve())
    while resolved in sys.path:
        sys.path.remove(resolved)


def check_s2_repo_exists() -> bool:
    """Check if the fish-speech S2 repo has been cloned."""
    return os.path.isdir(S2_REPO_DIR) and os.path.isfile(os.path.join(S2_REPO_DIR, "pyproject.toml"))


def check_s2_weights_exist() -> bool:
    """Check if S2 Pro model weights are downloaded."""
    required_files = ["config.json", "codec.pth"]
    has_weights = (
        os.path.isfile(os.path.join(S2_CHECKPOINT_DIR, "model.pth")) or
        any(f.endswith(".safetensors") for f in os.listdir(S2_CHECKPOINT_DIR) if os.path.isfile(os.path.join(S2_CHECKPOINT_DIR, f)))
    ) if os.path.isdir(S2_CHECKPOINT_DIR) else False
    has_required = all(
        os.path.isfile(os.path.join(S2_CHECKPOINT_DIR, f)) for f in required_files
    ) if os.path.isdir(S2_CHECKPOINT_DIR) else False
    return has_required and has_weights


def _patch_s2_repo():
    """Apply compatibility patches to the cloned S2 repo.

    Fixes upstream bug in reference_loader.py where
    `import torchaudio.io._load_audio_fileobj` inside __init__ creates a local
    binding for `torchaudio`, causing UnboundLocalError on torchaudio 2.9+
    (which removed list_audio_backends()).
    """
    ref_loader = os.path.join(
        S2_REPO_DIR, "fish_speech", "inference_engine", "reference_loader.py"
    )
    if not os.path.isfile(ref_loader):
        return
    try:
        with open(ref_loader, "r", encoding="utf-8") as f:
            content = f.read()
        # Only patch if the buggy pattern is present
        if "import torchaudio.io._load_audio_fileobj" in content:
            content = content.replace(
                "import torchaudio.io._load_audio_fileobj  # noqa: F401",
                "import importlib; importlib.import_module('torchaudio.io._load_audio_fileobj')  # noqa: F401",
            )
            with open(ref_loader, "w", encoding="utf-8") as f:
                f.write(content)
            print("🔧 Patched reference_loader.py (torchaudio 2.9+ compat)")
    except Exception as e:
        print(f"⚠️ Could not patch reference_loader.py: {e}")


def clone_s2_repo() -> Tuple[bool, str]:
    """Clone or update the latest fish-speech repo for S2 Pro support."""
    try:
        if check_s2_repo_exists():
            print("🔄 Updating fish-speech S2 repo...")
            result = subprocess.run(
                ["git", "pull"], cwd=S2_REPO_DIR,
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                _patch_s2_repo()
                print("✅ Fish Speech S2 repo updated")
                return True, "✅ Fish Speech S2 repo updated"
            else:
                print(f"⚠️ Git pull failed: {result.stderr}")
                return True, "✅ Fish Speech S2 repo exists (pull failed, using existing)"
        else:
            print("📥 Cloning fish-speech S2 repo...")
            result = subprocess.run(
                ["git", "clone", "--depth", "1", S2_REPO_URL, S2_REPO_DIR],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                _patch_s2_repo()
                print("✅ Fish Speech S2 repo cloned successfully")
                return True, "✅ Fish Speech S2 repo cloned successfully"
            else:
                print(f"❌ Failed to clone repo: {result.stderr}")
                return False, f"❌ Failed to clone repo: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "❌ Git operation timed out"
    except FileNotFoundError:
        return False, "❌ Git not found. Please install git."
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def download_s2_weights() -> Tuple[bool, str]:
    """Download S2 Pro model weights from HuggingFace using Python API."""
    try:
        if check_s2_weights_exist():
            print("✅ S2 Pro weights already present")
            return True, "✅ S2 Pro weights already downloaded"
        os.makedirs(S2_CHECKPOINT_DIR, exist_ok=True)
        print(f"📥 Downloading S2 Pro weights to {S2_CHECKPOINT_DIR}...")
        print("   This is ~11GB, it may take a while...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=S2_HF_REPO, local_dir=S2_CHECKPOINT_DIR,
                local_dir_use_symlinks=False,
            )
            if check_s2_weights_exist():
                print("✅ S2 Pro weights downloaded successfully")
                return True, "✅ S2 Pro weights downloaded successfully"
        except ImportError:
            print("⚠️ huggingface_hub not installed, trying CLI fallback...")
        except Exception as e:
            print(f"⚠️ snapshot_download failed: {e}, trying CLI fallback...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "huggingface_hub.commands.huggingface_cli",
                 "download", S2_HF_REPO, "--local-dir", S2_CHECKPOINT_DIR],
                capture_output=True, text=True, timeout=3600
            )
            if result.returncode == 0 and check_s2_weights_exist():
                print("✅ S2 Pro weights downloaded successfully")
                return True, "✅ S2 Pro weights downloaded successfully"
        except Exception:
            pass
        return False, (
            f"❌ Failed to download weights automatically.\n"
            f"Please run this command in your terminal:\n"
            f"  pip install huggingface_hub\n"
            f"  huggingface-cli download {S2_HF_REPO} --local-dir \"{S2_CHECKPOINT_DIR}\""
        )
    except Exception as e:
        return False, f"❌ Download error: {str(e)}"


def _import_s2_modules():
    """
    Import S2 fish-speech modules with proper isolation from S1.

    Both S1 and S2 use 'fish_speech' as the package name, so we:
    1. Snapshot sys.path before touching anything
    2. Temporarily remove S1 fish_speech modules from sys.modules
    3. Import S2 modules (+ capture the DAC class directly to avoid Hydra later)
    4. Restore S1 modules and revert sys.path to the snapshot
    """
    global FISH_S2_AVAILABLE, _s2_modules_imported, _pre_s2_sys_path

    if _s2_modules_imported:
        return True

    try:
        # ---- snapshot sys.path so we can undo pyrootutils pollution ----
        _pre_s2_sys_path = list(sys.path)

        # Save all existing fish_speech modules (S1)
        saved_modules = {}
        for key in list(sys.modules.keys()):
            if key == 'fish_speech' or key.startswith('fish_speech.'):
                saved_modules[key] = sys.modules.pop(key)

        _add_s2_to_path()

        try:
            # Clear OmegaConf 'eval' resolver that S1 registered
            try:
                from omegaconf import OmegaConf
                if OmegaConf.has_resolver("eval"):
                    OmegaConf.clear_resolver("eval")
            except Exception:
                pass

            # --- Import S2 modules (fish_speech now resolves to S2 repo) ---
            from fish_speech.inference_engine import TTSInferenceEngine as S2TTSEngine
            from fish_speech.models.text2semantic.inference import launch_thread_safe_queue as s2_launch_queue
            from fish_speech.utils.schema import ServeTTSRequest as S2TTSRequest
            from fish_speech.utils.schema import ServeReferenceAudio as S2RefAudio
            from fish_speech.utils.file import audio_to_bytes as s2_audio_to_bytes

            # Import DAC class directly so we can bypass Hydra when loading codec
            from fish_speech.models.dac.modded_dac import DAC as S2DAC
            from fish_speech.models.dac.modded_dac import ModelArgs as S2DACModelArgs

            # We still import dac.inference to make sure the module is cached for
            # anything that references it internally, but we will NOT use its
            # load_model (Hydra-based) function.
            import fish_speech.models.dac.inference  # noqa: F401

            # Store references in module globals
            globals()['S2TTSEngine'] = S2TTSEngine
            globals()['s2_launch_queue'] = s2_launch_queue
            globals()['S2TTSRequest'] = S2TTSRequest
            globals()['S2RefAudio'] = S2RefAudio
            globals()['s2_audio_to_bytes'] = s2_audio_to_bytes
            globals()['S2DAC'] = S2DAC
            globals()['S2DACModelArgs'] = S2DACModelArgs

            # Save the S2 fish_speech modules for later swaps
            s2_modules = {}
            for key in list(sys.modules.keys()):
                if key == 'fish_speech' or key.startswith('fish_speech.'):
                    s2_modules[key] = sys.modules[key]
            globals()['_s2_fish_modules'] = s2_modules

            FISH_S2_AVAILABLE = True
            _s2_modules_imported = True
            print("✅ Fish Speech S2 Pro modules imported successfully")
            return True

        finally:
            # Remove S2 fish_speech modules from sys.modules
            for key in list(sys.modules.keys()):
                if key == 'fish_speech' or key.startswith('fish_speech.'):
                    sys.modules.pop(key, None)

            # Restore S1 fish_speech modules
            sys.modules.update(saved_modules)

            # Revert sys.path to the pre-import snapshot.
            # This undoes both our _add_s2_to_path AND any pyrootutils.setup_root
            # side-effects that happened during the S2 imports.
            sys.path[:] = _pre_s2_sys_path

            # Re-register the eval resolver for S1
            try:
                from omegaconf import OmegaConf
                if not OmegaConf.has_resolver("eval"):
                    OmegaConf.register_new_resolver("eval", eval)
            except Exception:
                pass

    except ImportError as e:
        print(f"⚠️ Fish Speech S2 Pro import failed: {e}")
        traceback.print_exc()
        FISH_S2_AVAILABLE = False
        return False
    except Exception as e:
        print(f"⚠️ Fish Speech S2 Pro import error: {e}")
        traceback.print_exc()
        FISH_S2_AVAILABLE = False
        return False


def _swap_to_s2_modules():
    """Temporarily swap sys.modules to use S2 fish_speech modules.
    Also snapshots and adjusts sys.path.
    Returns (saved_s1_modules, saved_sys_path)."""
    saved_path = list(sys.path)

    saved = {}
    for key in list(sys.modules.keys()):
        if key == 'fish_speech' or key.startswith('fish_speech.'):
            saved[key] = sys.modules.pop(key)

    s2_mods = globals().get('_s2_fish_modules', {})
    sys.modules.update(s2_mods)
    _add_s2_to_path()

    # Clear OmegaConf eval resolver to avoid "already registered" errors
    try:
        from omegaconf import OmegaConf
        if OmegaConf.has_resolver("eval"):
            OmegaConf.clear_resolver("eval")
    except Exception:
        pass

    return saved, saved_path


def _restore_s1_modules(saved_tuple):
    """Restore S1 fish_speech modules after S2 operation.
    saved_tuple is (saved_s1_modules, saved_sys_path) from _swap_to_s2_modules."""
    saved, saved_path = saved_tuple

    # Remove S2 modules
    for key in list(sys.modules.keys()):
        if key == 'fish_speech' or key.startswith('fish_speech.'):
            sys.modules.pop(key, None)
    # Restore S1 modules
    sys.modules.update(saved)
    # Restore sys.path exactly
    sys.path[:] = saved_path

    # Re-register eval resolver for S1
    try:
        from omegaconf import OmegaConf
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)
    except Exception:
        pass


def _launch_s2_queue(checkpoint_path, device, precision, compile=True, max_seq_len=4096):
    """
    Our own version of launch_thread_safe_queue that caps max_seq_len
    for the KV cache.  The upstream S2 code uses model.config.max_seq_len
    (32768) which pre-allocates ~12 GB of KV cache alone.  For TTS,
    4096 tokens is more than enough and saves ~10 GB of VRAM.
    """
    import queue as _queue

    from fish_speech.models.text2semantic.inference import (
        GenerateRequest,
        WrappedGenerateResponse,
        generate_long,
        init_model,
    )

    input_queue = _queue.Queue()
    init_event = threading.Event()

    def worker():
        model, decode_one_token = init_model(
            checkpoint_path, device, precision, compile=compile
        )
        # ---- Cap the KV cache size ----
        capped = min(max_seq_len, model.config.max_seq_len)
        model.config.max_seq_len = capped
        print(f"   KV cache capped to max_seq_len={capped} (saves VRAM)")

        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=capped,
                dtype=next(model.parameters()).dtype,
            )
        model._cache_setup_done = True
        init_event.set()

        while True:
            item = input_queue.get()
            if item is None:
                break
            kwargs = item.request
            response_queue = item.response_queue
            try:
                for chunk in generate_long(
                    model=model, decode_one_token=decode_one_token, **kwargs
                ):
                    response_queue.put(
                        WrappedGenerateResponse(status="success", response=chunk)
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                import traceback as _tb
                from loguru import logger as _logger
                _logger.error(_tb.format_exc())
                response_queue.put(WrappedGenerateResponse(status="error", response=e))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()
    return input_queue


def _load_s2_codec(checkpoint_path: str, device: str = "cuda"):
    """
    Load the S2 DAC codec model WITHOUT Hydra.

    Hydra's instantiate() resolves '_target_: fish_speech.models.dac.modded_dac.DAC'
    via importlib, which can pick up the S1 package from sys.path (thanks to
    pyrootutils polluting it at startup).  We sidestep this entirely by using
    the S2 DAC class reference we captured during _import_s2_modules().

    The constructor args mirror fish_speech_s2/fish_speech/configs/modded_dac_vq.yaml.
    """
    from fish_speech.models.dac.rvq import DownsampleResidualVectorQuantize
    from fish_speech.models.dac.modded_dac import WindowLimitedTransformer

    transformer_config = S2DACModelArgs(
        block_size=2048, n_layer=8, n_head=16, dim=1024,
        intermediate_size=3072, n_local_heads=-1, head_dim=64,
        rope_base=10000, norm_eps=1e-5, dropout_rate=0.1,
        attn_dropout_rate=0.1, channels_first=True,
    )
    post_module = WindowLimitedTransformer(
        causal=True, window_size=128, input_dim=1024, config=transformer_config,
    )
    pre_module = WindowLimitedTransformer(
        causal=True, window_size=128, input_dim=1024, config=transformer_config,
    )
    quantizer = DownsampleResidualVectorQuantize(
        input_dim=1024, n_codebooks=9, codebook_size=1024, codebook_dim=8,
        quantizer_dropout=0.5, downsample_factor=[2, 2],
        post_module=post_module, pre_module=pre_module,
        semantic_codebook_size=4096,
    )
    # transformer_general_config must be a *callable* (functools.partial) because
    # EncoderBlock/DecoderBlock call it with extra kwargs like n_layer, n_head, dim.
    # Hydra achieves this with `_partial_: true`; we replicate it here.
    import functools
    general_config = functools.partial(
        S2DACModelArgs,
        block_size=8192, n_local_heads=-1, head_dim=64, rope_base=10000,
        norm_eps=1e-5, dropout_rate=0.1, attn_dropout_rate=0.1, channels_first=True,
    )
    model = S2DAC(
        sample_rate=44100,
        encoder_dim=64, encoder_rates=[2, 4, 8, 8],
        decoder_dim=1536, decoder_rates=[8, 8, 4, 2],
        encoder_transformer_layers=[0, 0, 0, 4],
        decoder_transformer_layers=[4, 0, 0, 0],
        quantizer=quantizer,
        transformer_general_config=general_config,
    )

    state_dict = torch.load(checkpoint_path, map_location=device, mmap=True, weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator" in k for k in state_dict):
        state_dict = {k.replace("generator.", ""): v for k, v in state_dict.items() if "generator." in k}

    result = model.load_state_dict(state_dict, strict=False, assign=True)
    model.eval()
    model.to(device)
    print(f"   DAC codec loaded: {result}")
    return model


def get_s2_status() -> str:
    """Get current status of Fish Speech S2 Pro."""
    lines = []
    lines.append("### Fish Speech S2 Pro Status\n")
    repo_ok = check_s2_repo_exists()
    weights_ok = check_s2_weights_exist()
    lines.append(f"- Repository: {'✅ Cloned' if repo_ok else '❌ Not cloned'}")
    lines.append(f"- Model Weights (~11GB): {'✅ Downloaded' if weights_ok else '❌ Not downloaded'}")
    lines.append(f"- Modules: {'✅ Imported' if _s2_modules_imported else '⭕ Not imported'}")
    lines.append(f"- Model: {'✅ Loaded' if FISH_S2_LOADED else '⭕ Not loaded'}")
    if not repo_ok:
        lines.append("\n**To get started:** Click 'Setup S2 Pro' to clone the repo and download weights.")
    elif not weights_ok:
        lines.append("\n**Next step:** Download model weights (~11GB).")
    elif not FISH_S2_LOADED:
        lines.append("\n**Ready to load.** Click 'Load' to start the model. Requires ~24GB VRAM.")
    return "\n".join(lines)


def setup_s2_pro(progress_callback=None) -> Tuple[bool, str]:
    """Full setup: clone repo, download weights."""
    print("=" * 50)
    print("🐟 Fish Speech S2 Pro - Setup Starting")
    print("=" * 50)
    messages = []
    print("\n📦 Step 1/2: Cloning fish-speech repo...")
    success, msg = clone_s2_repo()
    messages.append(msg)
    if not success:
        print("❌ Setup failed at Step 1")
        return False, "\n".join(messages)
    print("\n📦 Step 2/2: Downloading S2 Pro weights (~11GB, this may take a while)...")
    success, msg = download_s2_weights()
    messages.append(msg)
    if not success:
        print("❌ Setup failed at Step 2")
        return False, "\n".join(messages)
    print("\n" + "=" * 50)
    print("✅ Fish Speech S2 Pro - Setup Complete!")
    print("   You can now click 'Load' to start the model.")
    print("=" * 50)
    return True, "\n".join(messages)


def init_fish_speech_s2() -> Tuple[bool, str]:
    """Initialize Fish Speech S2 Pro model."""
    global FISH_S2_ENGINE, FISH_S2_LLAMA_QUEUE, FISH_S2_LOADED, FISH_S2_LOADING

    if FISH_S2_LOADED:
        return True, "✅ Fish Speech S2 Pro already loaded"
    if FISH_S2_LOADING:
        return False, "⏳ Fish Speech S2 Pro is currently loading..."
    if not check_s2_repo_exists():
        return False, "❌ Fish Speech S2 repo not cloned. Click 'Setup S2 Pro' first."
    if not check_s2_weights_exist():
        return False, "❌ S2 Pro weights not found. Click 'Setup S2 Pro' first."

    # Import modules if needed
    if not _s2_modules_imported:
        if not _import_s2_modules():
            return False, "❌ Failed to import S2 Pro modules. Check installation."

    try:
        FISH_S2_LOADING = True
        print("🔄 Loading Fish Speech S2 Pro (4B params)...")
        print("   This may take 30-60 seconds on first load...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        precision = torch.bfloat16  # S2 Pro requires bfloat16

        # Swap to S2 modules for the loading process
        saved = _swap_to_s2_modules()

        try:
            print("🔄 Loading LLAMA model and setting up inference queue...")
            FISH_S2_LLAMA_QUEUE = _launch_s2_queue(
                checkpoint_path=S2_CHECKPOINT_DIR,
                device=device,
                precision=precision,
                compile=False,
                max_seq_len=16384,  # Cap KV cache — ~20-22 GB VRAM on 4090
            )
            print("✅ LLAMA queue ready")

            # Load DAC codec WITHOUT Hydra to avoid S1 cross-contamination
            print("🔄 Loading DAC codec decoder (no Hydra)...")
            decoder_model = _load_s2_codec(
                checkpoint_path=os.path.join(S2_CHECKPOINT_DIR, "codec.pth"),
                device=device,
            )
            print("✅ DAC codec loaded")

            FISH_S2_ENGINE = S2TTSEngine(
                llama_queue=FISH_S2_LLAMA_QUEUE,
                decoder_model=decoder_model,
                precision=precision,
                compile=False,
            )
            print("✅ TTS inference engine created")
        finally:
            _restore_s1_modules(saved)

        FISH_S2_LOADED = True
        FISH_S2_LOADING = False
        print("=" * 50)
        print("✅ Fish Speech S2 Pro loaded successfully!")
        print("=" * 50)
        return True, "✅ Fish Speech S2 Pro loaded successfully"

    except Exception as e:
        FISH_S2_LOADING = False
        print(f"❌ Failed to load S2 Pro: {e}")
        traceback.print_exc()
        return False, f"❌ Failed to load S2 Pro: {str(e)}"


def unload_fish_speech_s2() -> str:
    """Unload Fish Speech S2 Pro to free memory."""
    global FISH_S2_ENGINE, FISH_S2_LLAMA_QUEUE, FISH_S2_LOADED
    try:
        print("🔄 Unloading Fish Speech S2 Pro...")
        if FISH_S2_ENGINE is not None:
            del FISH_S2_ENGINE
            FISH_S2_ENGINE = None
        if FISH_S2_LLAMA_QUEUE is not None:
            try:
                FISH_S2_LLAMA_QUEUE.put(None)
            except Exception:
                pass
            del FISH_S2_LLAMA_QUEUE
            FISH_S2_LLAMA_QUEUE = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        FISH_S2_LOADED = False
        print("✅ Fish Speech S2 Pro unloaded - VRAM freed")
        return "✅ Fish Speech S2 Pro unloaded - memory freed"
    except Exception as e:
        print(f"⚠️ Error unloading S2 Pro: {e}")
        return f"⚠️ Error unloading S2 Pro: {str(e)}"


def _split_text_chunks(text: str, max_chars: int = 300) -> list:
    """Split text into chunks, preferring sentence boundaries.
    
    Walks through the text and finds the last sentence-ending
    punctuation (.!?) within the max_chars window to break on.
    Falls back to last space if no punctuation found.
    Guarantees no characters are lost.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        # If remaining text fits, take it all
        if start + max_chars >= length:
            chunks.append(text[start:])
            break

        # Look for the last sentence-ending punctuation in the window
        window = text[start:start + max_chars]
        break_pos = -1

        # Search backwards for .!? followed by a space or bracket or end
        for i in range(len(window) - 1, -1, -1):
            if window[i] in '.!?' and (i + 1 >= len(window) or window[i + 1] in ' \n\t['):
                break_pos = i + 1  # include the punctuation
                break

        # Fallback: break at last space
        if break_pos <= 0:
            break_pos = window.rfind(' ')
            if break_pos <= 0:
                break_pos = max_chars  # no good break point, hard cut

        chunks.append(text[start:start + break_pos])
        start += break_pos

        # Skip leading whitespace on next chunk
        while start < length and text[start] in ' \t':
            start += 1

    return chunks if chunks else [text]


def generate_fish_s2_tts(
    text_input: str,
    ref_audio: str = None,
    ref_text: str = None,
    temperature: float = 0.8,
    top_p: float = 0.8,
    repetition_penalty: float = 1.1,
    max_tokens: int = 2048,
    seed: int = None,
    effects_settings=None,
    audio_format: str = "wav",
    skip_file_saving: bool = False,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    """
    Generate TTS audio using Fish Speech S2 Pro.

    S2 Pro supports inline emotion/prosody control via tags:
    e.g. "[excited] This is amazing! [laughing] Ha ha ha!"

    S2's generate_long() handles text batching internally using
    conversation history so the model maintains emotional context
    across chunks. We pass the full text and let S2 manage it.
    """
    if not FISH_S2_LOADED or FISH_S2_ENGINE is None:
        return None, "❌ Fish Speech S2 Pro not loaded - please load the model first"
    if not text_input or not text_input.strip():
        return None, "❌ No text provided"

    text_input = text_input.strip()
    print(f"\n🐟 S2 Pro - Generating speech for: {text_input[:80]}{'...' if len(text_input) > 80 else ''}")

    # Swap to S2 modules for the entire inference operation
    saved = _swap_to_s2_modules()

    try:
        # Prepare reference audio
        references = []
        if ref_audio and os.path.exists(ref_audio):
            try:
                ref_audio_bytes = s2_audio_to_bytes(ref_audio)
                references.append(S2RefAudio(audio=ref_audio_bytes, text=ref_text or ""))
                print(f"🐟 S2 Pro - Using reference audio: {os.path.basename(ref_audio)}")
            except Exception as e:
                print(f"⚠️ S2 Pro - Failed to load reference audio: {e}")

        request = S2TTSRequest(
            text=text_input,
            references=references,
            reference_id=None,
            format="wav",
            max_new_tokens=max_tokens,
            chunk_length=200,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            streaming=False,
            use_memory_cache="off",
            seed=seed,
            normalize=True,
        )

        print(f"🐟 S2 Pro - Starting inference (temp={temperature}, top_p={top_p}, max_tokens={max_tokens})...")
        t0 = time.time()

        final_audio = None
        sample_rate = None

        for result in FISH_S2_ENGINE.inference(request):
            if result.code == "error":
                error_msg = str(result.error) if result.error else "Unknown error"
                print(f"❌ S2 Pro inference error: {error_msg}")
                _restore_s1_modules(saved)
                return None, f"❌ S2 Pro error: {error_msg}"
            elif result.code == "final":
                sample_rate, final_audio = result.audio
                break

        elapsed = time.time() - t0

        if final_audio is None or sample_rate is None:
            print("❌ S2 Pro - No audio generated")
            _restore_s1_modules(saved)
            return None, "❌ No audio generated. Try different text or settings."

        total_dur = len(final_audio) / sample_rate
        print(f"✅ S2 Pro - Generated {total_dur:.1f}s of audio in {elapsed:.1f}s")

        # Restore S1 modules now that inference is done
        _restore_s1_modules(saved)

        # Ensure float32 for output
        if final_audio.dtype != np.float32:
            final_audio = final_audio.astype(np.float32)

        # Safety normalization
        peak = np.max(np.abs(final_audio))
        if peak > 1.0:
            final_audio = final_audio / peak
        elif peak > 0 and peak < 0.1:
            final_audio = final_audio / peak * 0.9

        # Apply effects if provided
        if effects_settings:
            try:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from launch import apply_audio_effects
                final_audio = apply_audio_effects(final_audio, sample_rate, effects_settings)
            except ImportError:
                pass

        status = f"✅ Generated with Fish Speech S2 Pro ({total_dur:.1f}s, {elapsed:.1f}s)"

        if not skip_file_saving:
            try:
                from launch import save_audio_with_format
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath, filename = save_audio_with_format(
                    final_audio, sample_rate, audio_format, "outputs", f"fish_s2_pro_{timestamp}"
                )
                status += f" - Saved as: {filename}"
            except Exception as e:
                print(f"⚠️ Could not save audio: {e}")

        return (sample_rate, final_audio), status

    except Exception as e:
        try:
            _restore_s1_modules(saved)
        except Exception:
            pass
        print(f"❌ S2 Pro generation error: {e}")
        traceback.print_exc()
        return None, f"❌ S2 Pro error: {str(e)}"




# ===== Handler accessor =====
def get_fish_s2_handler():
    """Get the singleton handler reference."""
    return {
        'available': FISH_S2_AVAILABLE or check_s2_repo_exists(),
        'loaded': FISH_S2_LOADED,
        'loading': FISH_S2_LOADING,
        'repo_exists': check_s2_repo_exists(),
        'weights_exist': check_s2_weights_exist(),
    }


# Try initial import if repo already exists
if check_s2_repo_exists():
    try:
        _import_s2_modules()
    except Exception:
        pass
