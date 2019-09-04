import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display


def show_spec(spec, sr, y_axis='log', scale_fn=librosa.amplitude_to_db, title=None, **kwargs):
    log_spec = scale_fn(spec, ref=np.max)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_spec, sr=sr, x_axis="time", y_axis=y_axis, **kwargs)
    plt.colorbar(format='%+2.0f dB')
    if title:
        plt.title(title)
