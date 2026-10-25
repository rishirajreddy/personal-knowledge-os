"""Wake-word detection using openWakeWord (local, Apache-2.0, no API key)."""
import subprocess
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
FRAME = 1280               # 80 ms, the chunk size openWakeWord expects
DEFAULT_WAKE_WORD = "alexa"   # built in: alexa, hey_mycroft, hey_jarvis, hey_rhasspy
DEFAULT_THRESHOLD = 0.5    # raise if it triggers by accident, lower if it misses you
CHIME = "/System/Library/Sounds/Tink.aiff"


class WakeWordDetector:
    def __init__(self, wake_word=DEFAULT_WAKE_WORD, threshold=DEFAULT_THRESHOLD):
        import openwakeword
        from openwakeword.model import Model
        openwakeword.utils.download_models([wake_word])  # no-op if already downloaded
        self.model = Model(wakeword_models=[wake_word], inference_framework="onnx")
        self.wake_word = wake_word
        self.threshold = threshold

    def wait(self) -> None:
        """Block until the wake word is heard."""
        self.model.reset()  # drop audio buffered from earlier turns
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                            blocksize=FRAME) as stream:
            while True:
                frame = stream.read(FRAME)[0][:, 0]
                scores = self.model.predict(frame)
                if max(scores.values()) >= self.threshold:
                    return


def chime() -> None:
    """Short non-blocking sound to show the assistant is listening."""
    try:
        subprocess.Popen(["afplay", CHIME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass

