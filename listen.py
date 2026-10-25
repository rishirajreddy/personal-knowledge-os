"""Hands-free listening: record one utterance, stopping when the user goes quiet."""
import collections
import time
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
BLOCK_SEC = 0.03
CALIBRATION_SEC = 0.5
END_SILENCE_SEC = 1.5    # silence that marks the end of an utterance
MIN_SPEECH_SEC = 0.15    # ignore clicks and coughs shorter than this
MAX_UTTERANCE_SEC = 30
PRE_ROLL_SEC = 0.3       # audio kept from before speech was detected
MIN_THRESHOLD = 0.01     # RMS floor so a silent room doesn't trigger on hiss


def _rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block ** 2)))


def listen_for_utterance(wait_timeout=None):
    """Block until the user speaks and finishes; return mono float32 audio.

    Returns an empty array if wait_timeout seconds pass with no speech
    (or if the detected sound was too short to be speech).
    """
    block_size = int(SAMPLE_RATE * BLOCK_SEC)
    pre_roll = collections.deque(maxlen=int(PRE_ROLL_SEC / BLOCK_SEC))
    frames = []
    speech_blocks = 0
    silent_blocks = 0
    speaking = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=block_size) as stream:
        # Measure the room's noise floor so the threshold adapts to the environment
        levels = [_rms(stream.read(block_size)[0][:, 0])
                  for _ in range(int(CALIBRATION_SEC / BLOCK_SEC))]
        threshold = max(float(np.median(levels)) * 3, MIN_THRESHOLD)

        start = time.time()
        while True:
            block = stream.read(block_size)[0][:, 0].copy()
            loud = _rms(block) > threshold

            if not speaking:
                pre_roll.append(block)
                if loud:
                    speaking = True
                    frames = list(pre_roll)
                    speech_blocks = 1
                elif wait_timeout and time.time() - start > wait_timeout:
                    return np.array([], dtype=np.float32)
                continue

            frames.append(block)
            if loud:
                speech_blocks += 1
                silent_blocks = 0
            else:
                silent_blocks += 1
            if silent_blocks * BLOCK_SEC >= END_SILENCE_SEC:
                break
            if len(frames) * BLOCK_SEC >= MAX_UTTERANCE_SEC:
                break

    if speech_blocks * BLOCK_SEC < MIN_SPEECH_SEC:
        print(f"[heard {speech_blocks * BLOCK_SEC:.2f}s of sound, too short - ignored]", flush=True)
        return np.array([], dtype=np.float32)
    return np.concatenate(frames)
