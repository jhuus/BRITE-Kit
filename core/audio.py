# Audio processing, especially extracting and returning spectrograms.

import logging
import warnings
warnings.filterwarnings('ignore') # librosa generates too many warnings

import cv2
import librosa
import numpy as np
import torch
import torchaudio as ta

from core import cfg

class Audio:
    def __init__(self, device='cuda'):
        self.have_signal = False
        self.path = None
        self.signal = None
        self.device = device

        self.linear_transform = ta.transforms.Spectrogram(
            n_fft=2*cfg.audio.win_length,
            win_length=cfg.audio.win_length,
            hop_length=int(cfg.audio.segment_len * cfg.audio.sampling_rate / cfg.audio.spec_width),
            power=1
        ).to(self.device)

        self.mel_transform = ta.transforms.MelSpectrogram(
            sample_rate=cfg.audio.sampling_rate,
            n_fft=2*cfg.audio.win_length,
            win_length=cfg.audio.win_length,
            hop_length=int(cfg.audio.segment_len * cfg.audio.sampling_rate / cfg.audio.spec_width),
            f_min=cfg.audio.min_audio_freq,
            f_max=cfg.audio.max_audio_freq,
            n_mels=cfg.audio.spec_height,
            power=cfg.audio.power,
            ).to(self.device)

    # width of spectrogram is determined by input signal length, and height = cfg.audio.spec_height
    def _get_raw_spectrogram(self, signal, segment_len):
        min_audio_freq = cfg.audio.min_audio_freq
        max_audio_freq = cfg.audio.max_audio_freq
        spec_height = cfg.audio.spec_height
        mel_scale = cfg.audio.mel_scale

        signal = signal.reshape((1, signal.shape[0]))
        tensor = torch.from_numpy(signal).to(self.device)
        if mel_scale:
            if segment_len == cfg.audio.segment_len:
                mel_transform = self.mel_transform
            else:
                mel_transform = ta.transforms.MelSpectrogram(
                    sample_rate=cfg.audio.sampling_rate,
                    n_fft=2*cfg.audio.win_length,
                    win_length=cfg.audio.win_length,
                    hop_length=int(segment_len * cfg.audio.sampling_rate / cfg.audio.spec_width),
                    f_min=cfg.audio.min_audio_freq,
                    f_max=cfg.audio.max_audio_freq,
                    n_mels=cfg.audio.spec_height,
                    power=cfg.audio.power,
                    ).to(self.device)

            spec = mel_transform(tensor).cpu().numpy()[0]
        else:
            if segment_len == cfg.audio.segment_len:
                linear_transform = self.linear_transform
            else:
                linear_transform = ta.transforms.Spectrogram(
                    n_fft=2*cfg.audio.win_length,
                    win_length=cfg.audio.win_length,
                    hop_length=int(segment_len * cfg.audio.sampling_rate / cfg.audio.spec_width),
                    power=1
                ).to(self.device)

            spec = linear_transform(tensor).cpu().numpy()[0]

        if not mel_scale:
            # clip frequencies above max_audio_freq and below min_audio_freq
            high_clip_idx = int(2 * spec.shape[0] * max_audio_freq / cfg.audio.sampling_rate)
            low_clip_idx = int(2 * spec.shape[0] * min_audio_freq / cfg.audio.sampling_rate)
            spec = spec[:high_clip_idx, low_clip_idx:]
            spec = cv2.resize(spec, dsize=(spec.shape[1], spec_height), interpolation=cv2.INTER_AREA)

        return spec

    # normalize values between 0 and 1
    def _normalize(self, specs):
        for i in range(len(specs)):
            if specs[i] is None:
                continue

            max = specs[i].max()
            if max > 0:
                specs[i] = specs[i] / max

            specs[i] = specs[i].clip(0, 1)

    # return list of spectrograms for the given offsets (i.e. starting points in seconds);
    # you have to call load() before calling this;
    # if raw_spectrograms array is specified, populate it with spectrograms before normalization
    def get_spectrograms(self, offsets, segment_len=None, low_band=False, raw_spectrograms=None):
        logging.debug(f"Audio::get_spectrograms offsets={offsets}")
        if not self.have_signal:
            return None

        if segment_len is None:
            # this is not the same as segment_len=cfg.audio.segment_len in the parameter list,
            # since cfg.audio.segment_len can be modified after the parameter list is evaluated
            segment_len = cfg.audio.segment_len

        specs = []
        sr = cfg.audio.sampling_rate
        for i, offset in enumerate(offsets):
            if int(offset*sr) < len(self.signal):
                spec = self._get_raw_spectrogram(self.signal[int(offset*sr):int((offset+segment_len)*sr)], segment_len)
                spec = spec[:cfg.audio.spec_height, :cfg.audio.spec_width]
                if spec.shape[1] < cfg.audio.spec_width:
                    spec = np.pad(spec, ((0, 0), (0, cfg.audio.spec_width - spec.shape[1])), 'constant', constant_values=0)
                specs.append(spec)
            else:
                specs.append(None)

        if raw_spectrograms is not None and len(raw_spectrograms) == len(specs):
            for i, spec in enumerate(specs):
                raw_spectrograms[i] = spec

        self._normalize(specs)

        return specs

    def signal_len(self):
        return len(self.signal) if self.have_signal else 0

    # if logging level is DEBUG, librosa.load generates a lot of output,
    # so temporarily update level
    def _call_librosa_load(self, path):
        saved_log_level = logging.root.level
        logging.root.setLevel(logging.ERROR)
        signal, sr = librosa.load(path, sr=cfg.audio.sampling_rate, mono=True)
        logging.root.setLevel(saved_log_level)

        return signal, sr

    def load(self, path):
        try:
            self.have_signal = True
            self.path = path
            self.signal, _ = self._call_librosa_load(path)

        except Exception as e:
            self.have_signal = False
            self.signal = None
            self.path = None
            logging.error(f'Caught exception in audio load of {path}: {e}')

        logging.debug('Done loading audio file')
        return self.signal, cfg.audio.sampling_rate
