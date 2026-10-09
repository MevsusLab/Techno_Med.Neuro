# TecnoMedNeuro Technical Reference

## 1. Project Purpose

TecnoMedNeuro is a Python desktop research prototype that uses a webcam and microphone to explore four explainable measurement workflows:

1. **Hand motion analysis:** MediaPipe hand landmarks followed by digital signal processing (DSP) of the index fingertip trajectory.
2. **Facial symmetry and movement analysis:** MediaPipe face landmarks measured at rest, during a smile, and during eyebrow elevation.
3. **Contactless rPPG pulse-rate estimation:** mean green-channel skin intensity followed by band-pass filtering, FFT peak analysis, and signal-quality checks.
4. **Acoustic voice measurement:** five-second sustained vowel, pitch, period/amplitude perturbation, harmonicity, and spectral features.

The educational objective is to connect computer vision, signal processing, numerical measurement, data-quality handling, process isolation, and a scientific desktop UI. The project is not a medical device, has no clinical validation, and must not be used to diagnose, exclude, or delay care for any condition. Eye-gaze AAC remains an interface mockup.

The facial module observes only the **Face** component of FAST. It does not assess arm weakness, speech, time of onset, or the complete FAST protocol. Sudden facial droop, arm weakness, or speech difficulty requires immediate emergency assessment regardless of software output.

## 2. Current Module Status

| Module | Implementation | Input | Output |
|---|---|---|---|
| TREMOR / Hand Motion | Working research prototype | Webcam, one hand | Image-plane motion RMS, exploratory dominant frequency, PSD, waveform, capture-quality information, and a protocol panel |
| FACIAL / Face Component of FAST | Working research prototype | Webcam, one face | Normalized bilateral geometry, smile and brow movement imbalance, expression-specific deviation map, and a protocol panel |
| rPPG Vitals | Working research prototype | Webcam, forehead and two cheeks | Estimated BPM, fractional intensity waveform, normalized FFT power, region estimates, and quality rejection |
| Eye-Tracking AAC | UI mockup only | None | No gaze tracking or communication is implemented |
| Voice Biomarkers | Working research prototype | Microphone, sustained vowel | F0, local/RAP jitter, local/APQ5 shimmer, HNR, waveform/pitch/spectrogram, MFCC summary and reports |

## 3. Runtime Architecture

```text
PyQt6 GUI process
  MainWindow / page widgets / Qt timers / Matplotlib Qt canvases
        |
        | spawn one isolated process per capture session
        | one-way data pipe: latest preview frame, scalar samples, phase, completion/error
        v
Capture supervisor (QThread in GUI process)
  - monitors child progress and phase deadlines
  - accepts cooperative cancellation
  - terminates and reaps a stuck child before allowing another session
        |
        v
Capture child process
  - owns OpenCV camera, MediaPipe task and inference lifecycle
  - samples acquisition timestamps and landmarks
  - releases camera/model in cleanup
```

### Why capture is isolated

Camera drivers and computer-vision inference execute native code. A Python thread can request cancellation, but cannot reliably interrupt a native camera read, model call, or resource destructor that never returns. The child-process boundary lets the supervisor recover from a hang or native crash without taking down the UI. `QThread.terminate()` is not used. One capture owns the camera at a time; the UI blocks navigation into another camera module while a capture is active. Child-to-parent data uses a one-way pipe; cancellation uses a shared multiprocessing Event.

### Capture message flow

`capture_worker.py` implements the shared `ProcessCaptureWorker` supervisor and hand-capture child loop. `face_capture.py` and `rppg_capture.py` supply child targets with the same packet contract. Packets transport the latest RGB preview frame, accumulated measurement samples, recording bounds, current phase and phase start time, status text, completion flag, error and exit reason. The GUI timer drains packets and updates widgets on the GUI thread. The preview keeps only the latest image to bound display memory; timestamped measurement samples are retained for analysis.

`voice_capture.py` also uses this supervisor. Its packets carry preparation/recording progress, microphone peak level, and the final acoustic result, not raw audio. Microphone opening has a 15-second phase timeout, reads have a 3-second timeout, and voice analysis has a 120-second timeout to accommodate first-use numerical/JIT initialization. Cooperative cancellation and forced process cleanup use the same existing supervisor path.

`voice_dependencies.py` checks module availability in the current interpreter before the VOICE UI starts a session. Failures display a PowerShell install command containing `sys.executable` and the absolute `requirements.txt` path. The child then resolves librosa's lazy feature imports, sounddevice, and Parselmouth before microphone access, under a separate 120-second `voice dependencies` timeout. Missing or broken analysis packages therefore fail before capture instead of after five seconds. Dependencies must be installed in the environment used to launch the app; the current local Python 3.14 `.venv` was found to lack librosa/Parselmouth even though system Python 3.12 had them. Voice analysis has also been verified in that `.venv` after installing the pinned requirements.

The supervisor uses Python multiprocessing `spawn`, phase-specific timeouts, cooperative stop, followed by process termination/kill if necessary. Each child creates and closes its own model and camera. Model downloads use a temporary file followed by rename; model URLs are version-pinned.

## 4. Technology Stack

### Python

The application and tests are Python. The documented runtime target is Python 3.12 on Windows. The entry point calls `multiprocessing.freeze_support()` before constructing the Qt application so spawned capture processes can start correctly on Windows.

### PyQt6 / Qt

PyQt6 provides the native desktop interface, layouts, buttons, status indicators, timers, `QThread`, child process supervisor integration, dialogs, and Qt PDF printing. The application uses a `QMainWindow`, horizontal technical navigation, and a `QStackedWidget` for the six module pages. UI updates and Matplotlib drawing are performed from the GUI thread.

### OpenCV (`cv2`)

OpenCV opens webcam device index 0, requests 640x480 at 30 FPS, reads frames, converts BGR camera frames to RGB for MediaPipe, draws landmark overlays, and converts captured frames into displayable images. Requested camera resolution/FPS are preferences; actual delivery rate is measured from acquisition timestamps and may differ.

### MediaPipe Tasks

The project uses the MediaPipe Tasks APIs (`HandLandmarker` and `FaceLandmarker`) in `VIDEO` mode with monotonically increasing millisecond timestamps. Models are loaded from local `.task` files, downloaded on first use if absent. Hand capture tracks one hand; facial capture permits up to two detected faces and accepts measurements only when exactly one face and a facial transform are available.

The current face model returns 478 landmarks in the verified runtime. Geometry uses the first 468 face landmarks; the remaining points are not used by the facial measurements. Face blendshapes are not enabled. The transform matrix output is used to estimate head pose for capture-quality rejection.

### NumPy

NumPy stores sample arrays, performs coordinate interpolation, vector geometry, medians and means, signal RMS, rotation/normalization operations, and finite-value validation.

### SciPy

`scipy.signal` supplies detrending, fourth-order Butterworth band-pass filters in second-order-section form, zero-phase filtering with `sosfiltfilt`, and Welch PSD estimates. `scipy.fft` supplies real-valued FFT magnitudes returned by the hand engine. The current TREMOR UI plots the combined Welch PSD and filtered waveform; it does not currently plot the returned FFT magnitude separately.

### Matplotlib

The QtAgg backend embeds plots in the PyQt UI. TREMOR uses Matplotlib for the live and completed waveforms and the combined-axis PSD. FACIAL uses scatter plots for normalized face landmarks and a fixed-range color map of selected regional landmark errors.

### Sounddevice / PortAudio

`sounddevice` uses the default input device, checks mono float32 capture at the device's native sample rate, and reads bounded blocks using `InputStream`. The five-second duration is determined by sample count, not timer labels. Input overflow or missing samples fail the session. PortAudio resources are closed before voice analysis starts.

### Librosa

Librosa supplies RMS framing, resampling to 16 kHz for analysis views, YIN fundamental-frequency estimation as an independent pitch sanity check, STFT, relative-dB spectrogram conversion, and 13 mean MFCC coefficients. It is not used to approximate standardized jitter, shimmer, or HNR from naive FFT peaks.

### Praat / Parselmouth

`praat-parselmouth` exposes Praat pitch autocorrelation, periodic cross-correlation pulse placement, local/RAP jitter, local/APQ5 shimmer, and autocorrelation harmonicity. These are established computational definitions but their presence does not clinically validate the prototype. There is no speech-to-text model or dysarthria classifier. `pyttsx3` would provide text-to-speech for a future AAC feature; it is not needed by acoustic voice capture and is not currently installed as a direct dependency.

### Additional modules

Python `multiprocessing`, `threading`, `urllib`, and `tempfile` provide process isolation, synchronization, and safe model downloads. `pathlib` is used for report output. The test suite uses `unittest`, Qt's offscreen platform, and mocks/synthetic signals; hardware-free tests do not require a webcam.

## 5. TREMOR: Hand Motion Measurement

### Capture protocol

1. The child process loads the pinned hand model and opens the camera.
2. The preview waits for one visible, stable index fingertip for two seconds; preparation has a 15-second timeout.
3. A 15-second measurement starts only after preparation succeeds.
4. The child records the actual camera-frame acquisition timestamp and the index fingertip's pixel coordinates. It captures a latest-frame preview independently from its ordered scalar measurement buffer.
5. The face/hand detector and camera are closed in child-process cleanup; final samples are drained before analysis.

The first camera frame has a 10-second startup allowance. After frames start, a read stall longer than two seconds is an error. Supervisor phase timeouts also cover model setup/initialization, camera opening, inference and cleanup.

### Signal analysis (`tremor_engine.py`)

The engine interface is `TremorEngine.analyze(record_start=None, record_end=None)`. The UI supplies the acquisition-time bounds; the engine verifies both bounds, recording duration and landmark coverage. Keeping this interface and result schema synchronized with the UI is important: an older `analyze(self)` implementation caused a previous end-of-scan crash.

Quality gates reject invalid/non-finite data, non-increasing timestamps, fewer than 200 hand samples, less than ten seconds of usable signal, median sample rate below 25 samples/s, coverage below 80%, a gap above 0.2 seconds, or missing hand samples more than 0.2 seconds from the recording boundaries. Short timing differences are interpolated onto a uniform time grid; longer outages are rejected rather than bridged.

Processing steps:

1. Estimate sample rate as the inverse median acquisition interval.
2. Interpolate X and Y trajectories to the median-rate time grid and remove linear trends independently.
3. Set the upper analysis band to `min(12 Hz, 0.45 * sample_rate, 0.4 / largest_gap_seconds)`. If less than 3 Hz remains, reject the recording. A reduced upper band is reported as `LIMITED BAND`.
4. Apply a fourth-order Butterworth 1-Hz-to-upper-band filter using SOS coefficients and zero-phase filtering.
5. Exclude the first and last second from RMS/spectral estimates to reduce edge transients.
6. Calculate Welch PSD independently for X and Y and add the powers. Select the maximum in-band bin as the candidate peak.
7. Calculate 2D motion RMS in pixels: `sqrt(mean(filtered_x**2 + filtered_y**2))` over the stable interior.
8. Calculate peak concentration as the fraction of in-band PSD power within +/-1 Hz of the candidate. This is spectral concentration, not diagnostic confidence.
9. Withhold the dominant-frequency result unless the exploratory peak-to-median ratio is at least 4, concentration is at least 30%, RMS is nontrivial, and the candidate is inside the usable band and away from the band edge.

The engine returns dominant frequency or `None`, RMS, concentration, sample rate, analyzed upper band, frequency resolution, coverage, largest gap, time series, and FFT/PSD arrays. Pixel motion depends on camera position, framing, detector noise and hand movement; it is not millimeters. Frequency ranges do not diagnose Parkinson's disease or essential tremor.

### TREMOR protocol panel

After valid analysis the left capture viewport splits between video and a scrollable research protocol panel. Reset, new capture and invalid results hide it. The panel offers text and PDF export using Qt dialogs/printing and includes quality metrics and limitations.

## 6. FACIAL: Face Symmetry and Movement

### Guided capture (`face_capture.py`)

The UI presents the emergency guidance persistently, then guides preparation followed by three timed stages:

| Stage | Timeline after stable preparation | Prompt |
|---|---:|---|
| Rest | 0-2 seconds prepare; 2-5 seconds capture | Relax the face |
| Smile | 5-7 seconds prepare; 7-10 seconds capture | Smile clearly and hold |
| Brows | 10-12 seconds prepare; 12-15 seconds capture | Raise both eyebrows and hold |

Preparation requires two seconds of valid, stable frontal face tracking and times out after 15 seconds. The sequence has a 40-second overall deadline. The model is configured for two faces so multiple faces can be rejected. The capture stores valid, unmirrored image coordinates and pose only in each measurement window. The preview draws the model tessellation, all 468 geometric points, the glabella-to-chin line and highlighted paired measurement points. The camera and model are isolated in the same supervised process framework as TREMOR.

Quality checks include one face only, valid face transform, eye-corner span of at least 90 px, yaw/pitch no more than 12 degrees, roll no more than 25 degrees, full-face bounds, luminance between 35 and 235, Laplacian blur variance at least 12, steady face center and scale during preparation, at most 20% rejected capture frames per expression, continuous timestamps, and consistent pose/distance across stages. Each stage window is three seconds and analysis requires at least 20 valid samples, at least two seconds of nominal stage-window duration, <=0.25-second leading/trailing gaps, and <=0.3-second internal gaps. The sampled valid timestamps must fall inside the stage window. The user must keep eyes open for measurement.

### Landmark normalization and symmetry (`facial_engine.py`)

1. Validate that the supplied array contains at least 468 finite landmarks and that pose is within its configured limits.
2. Use outer eye corners 33/263 as a scale and orientation reference. Translate the eye midpoint to the origin, rotate the eye line horizontal, and divide coordinates by eye-corner span. This corrects translation, scale and image-plane roll, but does not correct perspective distortion.
3. Fit a first-degree central axis using landmarks 9, 168, 1 and 152 (upper face/glabella region, nose bridge, nose tip and chin). The fit is `x = slope * y + offset`; degenerate or unstable fits are rejected.
4. Resolve each selected landmark into perpendicular distance from the axis and position along the axis. For each bilateral pair, mirror mismatch is `100 * hypot(abs(right_side)-abs(left_side), right_along-left_along)`, expressed relative to eye-corner span.
5. Regional errors are medians over selected pairs. Regions include mouth, eyes, brows and cheek geometry. The per-frame geometric aggregate is an equal-weight mean of those four regional errors.
6. For each stage, report medians across its accepted frames. The displayed ASI is the arithmetic mean of the rest, smile and brow stage aggregates. It is a custom normalized geometry statistic, not a clinical scale, probability, or percent of affected tissue.

The paired landmarks are explicitly listed in `facial_engine.py` (`PAIRS`). Cheek landmarks are only geometric proxies: this system does not measure nasolabial-fold depth. Eye aperture is vertical eyelid distance divided by eye width. Brow elevation references the outer eye. These definitions are intended to be understandable and testable, not asserted as validated clinical biomarkers.

### Movement and result visualization

Smile excursion is the 2D movement of each mouth corner from rest in normalized coordinates. Brow excursion is the absolute change in the two brow-elevation measurements. Each movement imbalance is `100 * abs(right-left) / (right+left)`. Scores are withheld if smile or brow movement is below the configured minimum; weak motion must not create a numerically unstable imbalance. A result is rejected if head pose or face scale shifts materially between the expression stages.

The plot selector shows rest, smile or brows. It overlays the normalized 468-point mesh, paired landmarks and fitted axis; pair errors use a fixed 0%-10% yellow-to-red scale. This is a geometric visualization, not a tissue/pathology heatmap. The right column displays ASI, smile/brow excursion imbalance, rest eye aperture, cheek proxy error and anatomical R/L axis distances.

### FACIAL protocol panel and threshold interpretation

After a valid session, the left video viewport splits approximately equally with the scrollable FACIAL protocol panel above the action buttons. `[X]`, reset and new capture hide it and restore full-height video. Text/PDF exports include the three metrics, stage frame/rate/reference-width metadata, the threshold explanation and disclaimer.

The requested 8%, 15%, 12% and 20% boundaries are **unvalidated exploratory bands**. They are not clinical cutoffs. The panel uses a priority order: elevated ASI or smile imbalance first (with brow value described only as an observed relationship); otherwise low measured deviation; otherwise moderate measured deviation. It avoids calling results FAST-positive/negative, stroke/TIA, central paresis, or Bell's palsy. A low measurement does not exclude a condition, and a high one does not establish one.

## 7. Desktop UI and Visualization

### rPPG pipeline (`rppg_capture.py`, `rppg_engine.py`)

rPPG uses the shared Face Landmarker asset and process supervisor. The camera requests 640x480 at 30 FPS. One frontal face and an outer-eye span of at least 90 px are required. Three convex face-landmark polygons are inset to 75% of their original spread: forehead `(10,109,108,151,337,338)`, right cheek `(50,101,205,187)`, and left cheek `(280,330,425,411)`. Each region must contain at least 80 pixels, remain within the image, and avoid excessive clipping or darkness. Mean RGB is extracted before overlay drawing. A background corner outside the detected face supplies a lighting-reference signal. This is not a skin-segmentation classifier; hair, makeup and shadows can contaminate the landmark polygons.

Preparation requires three seconds within a 3%-of-eye-span head-position tolerance and 3% scale tolerance. The trial captures 30 seconds using actual acquisition timestamps. Pose is limited to 12-degree yaw/pitch and 20-degree roll. The supervisor covers camera/model/inference/cleanup timeouts just as for the other modules. STOP discards a partial trial rather than presenting it as a completed measurement.

Analysis requires at least 20 seconds, 300 samples, median sampling rate >=15/s, >=85% coverage relative to both captured-frame count and median sampling rate, and no gaps or leading/trailing missing samples over 0.25 seconds. Motion uses the 5th-to-95th percentile range instead of a brittle maximum-minus-minimum gate. Robust center displacement over 0.5 eye spans, scale change over 0.25, or pose range over 20 degrees is rejected. The 95th percentile of approximately one-second differences rejects repeated rapid motion above 0.12 eye spans, 0.08 scale, or 8 degrees. Smaller slow drift exceeding the former 0.08/0.06/8-degree limits is a caution rather than automatic rejection; brief extreme outliers are also flagged. Such trials receive BPM only after the unchanged spectral, region-agreement, temporal-consistency, and nuisance-correlation checks pass. Green intensity jumps over 3% of mean intensity between samples are rejected as abrupt exposure changes. These motion limits are exploratory tolerances, not validated accuracy guarantees.

Green-channel averages are interpolated to uniform sampling, converted to percentage variation about each ROI mean, linearly detrended, and filtered using a fourth-order SOS Butterworth band-pass at 0.75-3 Hz with `sosfiltfilt`. First/last seconds are omitted. A Hann-windowed real FFT yields one-sided power. Candidate peaks are selected within 45-180 BPM, with log-power parabolic interpolation used to reduce bin quantization. Interpolation does not improve the physical frequency resolution.

At least two ROI peaks must agree within 0.10 Hz (6 BPM). Each used region requires >=45% in-band power within +/-0.15 Hz, peak-to-median power ratio >=8, and filtered modulation standard deviation >=0.005%. Used ROI power spectra are normalized and averaged. Estimates from both temporal halves must stay within 0.15 Hz (9 BPM) of the whole-trial estimate. Strong correlation (>0.8) between the candidate waveform and filtered head-motion or background intensity signals rejects it when the nuisance signal has nontrivial amplitude.

Results include BPM, frequency, spectral concentration, median sampling rate, coverage, largest gap, bin spacing, each ROI's candidate BPM, used-region indices, filtered fractional waveform, normalized FFT power, and motion cautions. Only after a valid completed trial does the UI show BPM; otherwise it shows a specific RETRY reason. When drift/outliers were observed, the UI uses amber CHECK QUALITY and ESTIMATE / CAUTION badges and includes the warning in the results text. These gates are exploratory, not validated error bounds. Shared lighting artifacts, harmonic ambiguity, and camera image processing can still generate an incorrect accepted estimate. This module provides no SpO2, blood-pressure or arrhythmia measurement.

The rPPG completed-result panel shares `ProtocolPanel` layout/export code with TREMOR and FACIAL. `RPPGProtocolPanel` displays the Vitals Screening title, SAVE REPORT / EXPORT PDF / X controls, and its module-specific disclaimer. A zero-margin vertical layout with equal stretch factors places video and panel in separate non-overlapping rows above the capture controls. Hidden panels consume no layout space; reset/new capture/error clears the report, and X hides the panel. Completed caution estimates also reveal the panel without removing their warnings.

`rppg_protocol_text` evaluates quality first: coverage below 70%, concentration below 60%, or maximum-minus-minimum of all three ROI candidates above 5 BPM withholds protocol rate interpretation. Otherwise it describes estimates below 60, within 60-95 inclusive, or above 95 BPM. These thresholds are not clinical cutoffs. The panel explicitly notes that all-region spread includes excluded ROIs, whereas engine agreement is based on used ROIs; the stricter panel evaluation does not retroactively change the engine's numeric candidate. Reports identify candidate values, regions used, recording metadata, warnings, and limitations. They do not infer normal hemodynamics, autonomic equilibrium, conduction disease, or cause of a high/low rate.

`app.py` assembles the application. `MainWindow` owns the top navigation, page stack, shared hardware/DSP status bar and lifecycle-aware close handling. `TremorPage` and `FacialPage` own their session state, buttons, camera preview, metrics and plots. A shared `VideoLabel` stores the original frame and rescales it with `KeepAspectRatio` on resize. `PlotCanvas` embeds a Matplotlib `FigureCanvasQTAgg` and applies the engineering palette. The interface uses system sans-serif for headings, Consolas for technical values/status, square geometry, thin separators and neutral state badges.

The FACIAL, TREMOR and rPPG capture processes are mutually exclusive through navigation guards. `QTimer` drains process messages without blocking the Qt event loop. Process completion slots use `finally`-style cleanup so timers, worker references, buttons and close signals are restored even if analysis raises.

## 8. Data Handling and Privacy

The current capture pipelines do not write camera video to disk. Frames travel in memory between child process and GUI for live preview; scalar landmark samples are held only for the current analysis session. The user may explicitly export a report. Model assets are stored in `models/`. The `.task` models are downloaded on first use unless already installed. Reports currently contain derived measurements and quality information rather than raw video.

Voice audio is held in the child process for one trial and is not automatically saved. Derived pitch/waveform/spectrogram arrays and metrics are sent to the UI. Text/PDF export contains summaries and method settings, not an audio recording.

### Voice pipeline and metric definitions

The VOICE UI supports microphone recording and file upload. Microphone capture uses the default device as mono float32 at the native rate (16-96 kHz); reads are blocks of up to 1024 samples. A file session uses librosa's decoder, is limited to ten minutes/100 MB, is mixed to mono, and is resampled to 16 kHz only if its source rate falls outside the supported range. For a file longer than five seconds, a sliding RMS activity search selects a five-second candidate segment; the segment offset and source-file name are included in the exported report. A file shorter than 4.95 seconds or without a continuous voiced segment is rejected. Codec handling for MP3/M4A/AAC depends on the audio backend, so WAV/FLAC/OGG are the most portable inputs. File decoding and analysis run in the isolated child-process supervisor just like microphone capture. `VoiceFileCaptureWorker` passes the selected path as a spawn-safe process argument; file analysis uses the same `analyze_voice` gates and report schema as microphone capture.

`VoiceSession` captures exactly five seconds of mono float32 audio at the default device sample rate. Reads use blocks of at most 1024 samples. Overflow, missing blocks, cancellation, or native failures prevent a completed result. `analyze_voice(audio, sample_rate)` executes in the child process.

The microphone five-second buffer starts at detected sound onset instead of microphone startup. Before onset, 1024-sample blocks are monitored for RMS >=0.005; at least 80 ms of consecutive above-threshold blocks is needed. These onset blocks are retained as the start of the recording; earlier silence is discarded. Ten seconds without detection produces an explicit microphone/input-level error. This is amplitude-based detection, not recognition of a vowel, and does not change subsequent phonation-quality gates. Cancellation during onset waiting closes the input stream without analyzing a partial buffer. Uploaded files use the corresponding five-second selected segment rather than the voice-onset wait.

Recording-quality gates require a duration in 4.95-5.1 seconds, finite mono samples, clipping fraction <=0.1% at absolute amplitude 0.99, DC-removed RMS level >=-45 dBFS, and at least 85% active RMS frames. The activity floor is the larger of -45 dBFS and 10% of the recording's 90th-percentile RMS. RMS uses approximately 30 ms frames with 10 ms hops. These are engineering limits, not population-calibrated clinical thresholds.

The first/last 0.3 seconds are excluded. Praat autocorrelation pitch uses 75-500 Hz limits and a 10 ms time step. At least 80% voiced pitch frames and 100 usable frames are required. Median Praat F0 must agree within 15% with the median of librosa YIN estimates on the 16 kHz analysis copy; disagreement can indicate octave error or noise and is rejected.

Praat periodic cross-correlation pulse positions drive perturbation measurement. Period bounds are `0.8/500` and `1.25/75` seconds, maximum period factor 1.3, and maximum amplitude factor 1.6. Local jitter is average absolute consecutive-period difference divided by mean period. RAP compares each period against its three-period neighborhood. Local shimmer is average absolute consecutive-cycle amplitude difference divided by mean amplitude; APQ5 uses a five-cycle neighborhood. Percentages are multiplied by 100, with undefined values rejected. HNR is the mean of valid Praat autocorrelation harmonicity frames (10 ms step, minimum pitch 75 Hz, silence threshold 0.1, 4.5 periods/window); invalid sentinel values are removed.

STFT uses 1024-point frames and 160-sample hops at 16 kHz. The spectrogram is displayed at -80 to 0 dB relative to that trial's maximum, up to 5 kHz; this is not sound pressure level. Thirteen mean MFCC values are exported as features, with no learned classifier attached. Plots include the analyzed waveform, pitch trajectory, and spectrogram.

The report panel compares local jitter >1.04% or local shimmer >3.81% only with legacy MDVP references documented in the [Praat jitter manual](https://praat.org/manual/Voice_2__Jitter.html) and [shimmer manual](https://praat.org/manual/Voice_3__Shimmer.html). The manual notes noise/method dependence. RAP/APQ5 are named separately and are not compared with the local thresholds. No universal HNR threshold is imposed. A sustained vowel measures phonation and cannot establish dysarthria, dysphonia, or normal articulation. Microphone gain, automatic processing, room noise, pitch range and tracking failures can affect the measurements.

### Russian user-facing text

`i18n_ru.py` supplies `ru()` and `RussianLabel`. Module headings, action buttons, live prompts, post-test explanatory text, protocol actions, and disclaimers use Russian. Fixed string mappings and specific dynamic templates preserve numeric values; unknown technical/library messages fall back to their original text. `make_status_badge` deliberately uses a regular QLabel so internal status codes remain unchanged. Metric keys, state enums, model labels and Matplotlib chart text are not globally translated. Protocol exports translate narrative text at output time and retain method names/numeric metadata. Tests verify Russian headings, dynamic prompts, post-test branches, disclaimers, and unaffected measurement behavior.

## 9. Dependencies and Installation

Exact direct runtime versions are pinned in `requirements.txt`:

| Package | Role |
|---|---|
| `numpy==2.4.3` | Arrays, coordinate geometry, interpolation and summary statistics |
| `scipy==1.18.1` | Filters, detrending, Welch PSD and FFT |
| `opencv-contrib-python==5.0.0.93` | Webcam I/O, image conversion and overlays; contrib build satisfies MediaPipe's dependency |
| `mediapipe==1.0.1` | Tasks API hand and face landmark models |
| `PyQt6==6.11.0` | Desktop widgets, process supervisor threads, dialogs and printing |
| `matplotlib==3.11.1` | Embedded Qt plots |
| `sounddevice==0.5.6` | PortAudio microphone input |
| `librosa==1.0.0` | RMS, YIN sanity check, resampling, STFT and MFCC features |
| `praat-parselmouth==0.4.7` | Praat pitch, pulse/perturbation algorithms and HNR |

From the project directory on Python 3.12:

```powershell
python -m pip install -r requirements.txt
python app.py
```

The first session that needs a model requires network access to the pinned Google-hosted MediaPipe asset. Webcam access must be granted to the Python process. In Windows terminals, run the application as a script (`python app.py`) so multiprocessing `spawn` can import its top-level child-process targets. `freeze_support()` is called by the entry point for packaged Windows execution.

## 10. Testing

Run the complete suite from the project directory:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
python -m unittest discover -p "test_*.py"
```

Tests cover synthetic tremor sinusoids/noise/timing gaps, recording-bound checks, repeated UI sessions, worker cancellation and cleanup, mocked native hangs/crashes, real hand/face model initialization in child processes, facial symmetry/pose/expression metrics, failed/rejected sessions, viewport split geometry, report text/PDF output, and protocol-threshold boundaries. These tests are engineering/regression tests, not clinical validation. Webcam-specific performance, lighting robustness, test-retest reliability, and comparison with clinically assessed reference data remain to be evaluated.

## 11. Source Map

| File | Responsibility |
|---|---|
| `app.py` | PyQt application, page layouts, session state, charts, dashboard and UI-to-engine integration |
| `capture_worker.py` | Hand camera/MediaPipe session, shared child-process supervisor and transfer protocol |
| `tremor_engine.py` | Hand trajectory validation, filtering, PSD/FFT and derived metrics |
| `face_capture.py` | Face model download, face camera session, prompts, quality checks, overlays and face child-process transport |
| `facial_engine.py` | Face pose gates, axis normalization, regional symmetry and expression movement metrics |
| `rppg_capture.py` | Face-landmark skin polygons, mean RGB/background/motion sampling, and isolated capture |
| `rppg_engine.py` | Green-channel filtering, FFT pulse candidates, ROI agreement and nuisance checks |
| `voice_capture.py` | Isolated microphone and audio-file analysis sessions/result transport |
| `voice_dependencies.py` | Interpreter-specific availability checks, lazy import verification, and actionable installation errors |
| `voice_engine.py` | Audio quality gates, Praat acoustic metrics and librosa features |
| `i18n_ru.py` | Russian headings, instructions, dynamic guidance and protocol localization |
| `protocol_panel.py` | Shared report-panel styling/export plus hand/facial protocol text generation |
| `test_tremor_engine.py` | Synthetic DSP regression suite |
| `test_capture_ui.py` | Hand capture lifecycle and UI regression tests |
| `test_process_capture.py` | Child-process transport, timeout recovery and real model lifecycle tests |
| `test_facial_engine.py` | Synthetic facial geometry, pose and expression tests |
| `test_face_capture.py` | Face session, cancellation, quality and FACIAL UI tests |
| `test_protocol_panel.py` | Hand protocol-panel layout/content/export tests |
| `test_facial_protocol.py` | Facial split layout, threshold text, visibility and report export tests |
| `test_rppg_engine.py` | Known pulse rates, noise, sampling gaps, exposure and motion/lighting confounders |
| `test_rppg_capture.py` | ROI color extraction, guided capture, UI reset/retry, repeatability and camera ownership |
| `test_rppg_protocol.py` | Completed/caution split layout, close/reset visibility, quality/rate thresholds and text/PDF report exports |
| `test_voice.py` | Synthetic pitch, noise/perturbation, input quality, microphone lifecycle, UI cancellation and exports |
| `test_voice_dependencies.py` | Missing/broken packages and no-microphone preflight failures in both GUI and child |
| `test_i18n_ru.py` | Russian module titles, prompts, numeric preservation and narrative branch coverage |
