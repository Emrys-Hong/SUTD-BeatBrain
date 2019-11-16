import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa


def show_spec(spec, scale_fn=librosa.power_to_db, title=None, clean=True, **kwargs):
    """
    Display a spectrogram

    Args:
        spec (np.ndarray): The spectrogram to display
        scale_fn (function): A function that scales the spectrogram. If false, no scaling is performed
        title (str): The title of the plot
        clean (bool): If True, removes axis labels, colorbar, etc.
        **kwargs (dict): Additional keyword arguments passed to `sns.heatmap`
    """
    if scale_fn:
        spec = scale_fn(spec, ref=np.max)
    kwargs["cmap"] = kwargs.get("cmap") or "magma"
    sns.heatmap(
        spec[::-1],
        cbar_kws={"format": "%+2.0f dB"},
        xticklabels=not clean,
        yticklabels=not clean,
        cbar=not clean,
        **kwargs
    )
    plt.title(title)
