import numpy as np
import soundfile as sf
from transformers import pipeline

pipe = pipeline("text-to-speech", model="suno/bark-small")
text = "[clears throat] This is a test ... and I just took a long pause."
output = pipe(text)

rate = int(output["sampling_rate"])
audio = output["audio"].squeeze()

# 2. Convert the audio data to a 16-bit integer format (standard for WAV)
# Bark often outputs float32, and for wavfile.write to work correctly and
# to ensure standard playback compatibility, it's best to normalize and convert.
# We multiply by 32767 (max value for 16-bit signed integer) and convert to int16.
audio_int16 = (audio * 32767).astype(np.int16)

# 3. Use scipy.io.wavfile.write to save the file.
output_filename = "test_speech.wav"
sf.write(output_filename, audio, rate, subtype="PCM_16", format="WAV")
