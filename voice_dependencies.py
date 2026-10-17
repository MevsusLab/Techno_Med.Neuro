import importlib
import importlib.util
import sys
from pathlib import Path


voice_packages = {
    'sounddevice': 'sounddevice',
    'librosa': 'librosa',
    'parselmouth': 'praat-parselmouth',
}
voice_file_packages = {
    'librosa': 'librosa',
    'parselmouth': 'praat-parselmouth',
}

# для совместимости
VOICE_PACKAGES = voice_packages
VOICE_FILE_PACKAGES = voice_file_packages


def dependency_message(packages, cause=''):
    requirements = Path(__file__).resolve().parent / 'requirements.txt'
    executable = sys.executable.replace("'", "''")
    manifest = str(requirements).replace("'", "''")
    message = (
        f"Модуль голоса недоступен: проверьте зависимости {', '.join(packages)}.\n"
        f"Запущенный Python: {sys.executable}\n"
        f"Выполните в PowerShell:\n& '{executable}' -m pip install -r '{manifest}'\n"
        'После установки перезапустите приложение.'
    )
    if cause:
        message += '\nПодробности: ' + cause
    return message


def check_voice_dependencies():
    # смотрим установлены ли библиотеки для микрофона
    missing = []
    for module, package in voice_packages.items():
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            available = False
        if not available:
            missing.append(package)
    if missing:
        raise RuntimeError(dependency_message(missing))


def load_voice_dependencies():
    check_voice_dependencies()
    try:
        sounddevice = importlib.import_module('sounddevice')
        librosa = importlib.import_module('librosa')
        parselmouth = importlib.import_module('parselmouth')
        importlib.import_module('parselmouth.praat')
        _ = (sounddevice.InputStream, librosa.feature.rms, librosa.yin,
             librosa.resample, librosa.stft, librosa.feature.mfcc, parselmouth.Sound)
    except Exception as exc:
        raise RuntimeError(dependency_message(voice_packages.values(), f'{type(exc).__name__}: {exc}')) from exc


def check_voice_file_dependencies():
    # проверка пакетов для анализа файла
    missing = []
    for module, package in voice_file_packages.items():
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            available = False
        if not available:
            missing.append(package)
    if missing:
        raise RuntimeError(dependency_message(missing))


def load_voice_file_dependencies():
    # импорт либросы и праата без звуковой карты
    check_voice_file_dependencies()
    try:
        librosa = importlib.import_module('librosa')
        parselmouth = importlib.import_module('parselmouth')
        importlib.import_module('parselmouth.praat')
        _ = (librosa.feature.rms, librosa.yin, librosa.resample, librosa.stft,
             librosa.feature.mfcc, parselmouth.Sound)
    except Exception as exc:
        raise RuntimeError(dependency_message(voice_file_packages.values(),
                                              f'{type(exc).__name__}: {exc}')) from exc
