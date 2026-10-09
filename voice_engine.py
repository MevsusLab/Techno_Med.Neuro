import numpy as np
import os


PITCH_FLOOR = 75.0
PITCH_CEILING = 500.0


def load_audio_file(path):
    import librosa
    import soundfile as sf

    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise ValueError(f'Could not decode this audio file ({type(exc).__name__}: {exc}). Try WAV, FLAC, or OGG.') from exc
    if info.frames <= 0 or info.frames / info.samplerate > 600 or info.frames * info.channels * 4 > 100_000_000:
        raise ValueError('Audio file is empty or exceeds the 10-minute / 100 MB analysis limit.')
    try:
        audio, sample_rate = librosa.load(str(path), sr=None, mono=True)
    except Exception as exc:
        raise ValueError(f'Could not decode this audio file ({type(exc).__name__}: {exc}). Try WAV, FLAC, or OGG.') from exc
    if sample_rate < 16000 or sample_rate > 96000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    if len(audio) / sample_rate < 4.95:
        raise ValueError(f'Audio file is only {len(audio) / sample_rate:.2f} seconds. A five-second vowel segment is required.')
    if len(audio) <= sample_rate * 5.1:
        start = 0
    else:
        frame_length = int(sample_rate * 0.03)
        hop = int(sample_rate * 0.01)
        envelope = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop, center=False)[0]
        threshold = max(10 ** (-45 / 20), np.percentile(envelope, 90) * 0.1)
        active = (envelope > threshold).astype(float)
        window_frames = int(4.95 * sample_rate / hop)
        scores = np.convolve(active, np.ones(window_frames) / window_frames, mode='valid')
        start = int(np.argmax(scores) * hop)
        if scores[start // hop] < 0.75:
            raise ValueError('No continuous five-second voiced segment found. Use a file with one sustained Ah vowel.')
    segment = audio[start:start + sample_rate * 5]
    if len(segment) / sample_rate < 4.95:
        raise ValueError('Could not select a complete five-second vowel segment.')
    return segment, int(sample_rate), {
        'source': 'file', 'file_name': os.path.basename(str(path)), 'segment_start_sec': start / sample_rate,
        'source_sample_rate': int(info.samplerate), 'source_duration_sec': info.frames / info.samplerate,
    }


def analyze_voice(audio, sample_rate):
    import librosa
    import parselmouth
    from parselmouth.praat import call

    y = np.asarray(audio, dtype=np.float64)
    if y.ndim != 1 or not np.isfinite(y).all() or not 16000 <= sample_rate <= 96000:
        raise ValueError('Invalid mono audio or sample rate.')
    duration = len(y) / sample_rate
    if not 4.95 <= duration <= 5.1:
        raise ValueError('A complete five-second recording is required.')
    clipping = float(np.mean(np.abs(y) >= 0.99))
    if clipping > 0.001:
        raise ValueError(f'Audio is clipped ({clipping * 100:.2f}%). Reduce microphone gain or move farther away.')
    y = y - np.mean(y)
    rms = float(np.sqrt(np.mean(y ** 2)))
    level = 20 * np.log10(max(rms, 1e-12))
    if level < -45:
        raise ValueError(f'Recording is too quiet ({level:.1f} dBFS). Check microphone input and speak at a comfortable steady volume.')
    frame_length = max(256, int(sample_rate * 0.03))
    hop = int(sample_rate * 0.01)
    envelope = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop, center=False)[0]
    active = envelope > max(10 ** (-45 / 20), np.percentile(envelope, 90) * 0.1)
    if np.mean(active) < 0.85:
        raise ValueError('Too much silence or interruption. Sustain one continuous Ah vowel for all five seconds.')
    trim = int(0.3 * sample_rate)
    signal = y[trim:-trim]
    sound = parselmouth.Sound(signal, sampling_frequency=sample_rate)
    pitch = sound.to_pitch_ac(time_step=0.01, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    frequencies = pitch.selected_array['frequency']
    voiced = frequencies > 0
    voiced_fraction = float(np.mean(voiced))
    if voiced_fraction < 0.8 or voiced.sum() < 100:
        raise ValueError(f'Only {voiced_fraction * 100:.0f}% of frames have usable periodic voicing. Repeat a continuous vowel; irregular voices may be unmeasurable by this method.')
    f0 = float(np.median(frequencies[voiced]))
    acoustic = librosa.resample(signal, orig_sr=sample_rate, target_sr=16000)
    yin = librosa.yin(acoustic, fmin=PITCH_FLOOR, fmax=PITCH_CEILING, sr=16000, frame_length=1024, hop_length=160)
    if abs(np.median(yin) / f0 - 1) > 0.15:
        raise ValueError('Pitch estimators disagree; possible octave error, noise, or irregular phonation. No stable metric set can be reported.')
    pulse = call(sound, 'To PointProcess (periodic, cc)', PITCH_FLOOR, PITCH_CEILING)
    shortest, longest = 0.8 / PITCH_CEILING, 1.25 / PITCH_FLOOR
    jitter = float(call(pulse, 'Get jitter (local)', 0, 0, shortest, longest, 1.3) * 100)
    rap = float(call(pulse, 'Get jitter (rap)', 0, 0, shortest, longest, 1.3) * 100)
    shimmer = float(call([sound, pulse], 'Get shimmer (local)', 0, 0, shortest, longest, 1.3, 1.6) * 100)
    apq5 = float(call([sound, pulse], 'Get shimmer (apq5)', 0, 0, shortest, longest, 1.3, 1.6) * 100)
    harmonicity = sound.to_harmonicity_ac(time_step=0.01, minimum_pitch=PITCH_FLOOR, silence_threshold=0.1, periods_per_window=4.5)
    harmonic_values = harmonicity.values.ravel()
    harmonic_values = harmonic_values[np.isfinite(harmonic_values) & (harmonic_values > -199)]
    hnr = float(np.mean(harmonic_values)) if len(harmonic_values) else float('nan')
    if not np.isfinite([f0, jitter, rap, shimmer, apq5, hnr]).all():
        raise ValueError('Cycle-based voice metrics are undefined for this recording. No clinical inference can be made.')
    spectrum = np.abs(librosa.stft(acoustic, n_fft=1024, hop_length=160))
    spectrogram = librosa.amplitude_to_db(spectrum, ref=np.max)
    mfcc = librosa.feature.mfcc(y=acoustic, sr=16000, n_mfcc=13, n_fft=1024, hop_length=160)
    warnings = []
    if level > -10:
        warnings.append('High recording level; keep microphone distance and gain consistent.')
    if voiced_fraction < 0.95:
        warnings.append('Some analyzed frames are unvoiced; perturbation metrics describe accepted cycles only.')
    if np.mean(frequencies[voiced] <= PITCH_FLOOR * 1.1) > 0.1 or np.mean(frequencies[voiced] >= PITCH_CEILING * 0.9) > 0.1:
        warnings.append('Pitch is near the 75-500 Hz analysis limits; interpretation is limited.')
    return {
        'f0': f0, 'jitter_local': jitter, 'jitter_rap': rap, 'shimmer_local': shimmer,
        'shimmer_apq5': apq5, 'hnr': hnr, 'voiced_fraction': voiced_fraction,
        'level_dbfs': level, 'clipping': clipping, 'duration': duration, 'sample_rate': sample_rate,
        'pitch_time': pitch.xs() + 0.3, 'pitch': np.where(voiced, frequencies, np.nan),
        'wave_time': np.arange(len(acoustic)) / 16000 + 0.3, 'wave': acoustic,
        'spectrogram': spectrogram, 'spectrogram_time': np.arange(spectrogram.shape[1]) * 0.01 + 0.3,
        'spectrogram_freq': librosa.fft_frequencies(sr=16000, n_fft=1024), 'warnings': warnings,
        'mfcc_mean': np.mean(mfcc, axis=1).tolist(),
    }
