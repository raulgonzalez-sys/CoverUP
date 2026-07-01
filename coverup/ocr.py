"""
Optional OCR support for CoverUP PDF, backed by Tesseract.

Tesseract is an *optional* runtime dependency: the binary is not bundled, so
every entry point here degrades gracefully when it is missing. Callers should
gate any OCR UI on :func:`is_available` and disable it (with a hint to install
``tesseract-ocr``) otherwise.

Language data is resolved **on demand** (see :func:`ensure_languages`):
CoverUP keeps its own ``tessdata`` directory under the user data dir and makes
it self-sufficient. English ships as the base; any other language — typically
the one matching the system locale — is copied from the system/bundle if
present, or otherwise downloaded (a single ~1-3 MB ``.traineddata`` from the
``tessdata_fast`` project) the first time it is needed. All OCR then runs with
``--tessdata-dir`` pointed at that single directory, so mixed system/downloaded
languages always resolve.

Two features build on this module:

* **Auto-redaction of scanned pages** — :func:`find_boxes_ocr` runs OCR on a
  page image and reuses :func:`coverup.textsearch.match_units`, so the
  pattern/keyword matching is identical to the digital-PDF path.
* **Searchable export** — :func:`ocr_words` returns word boxes that
  ``main.py`` lays down as an invisible text layer over the *already redacted*
  page image, making the exported PDF searchable without re-introducing any
  redacted text (it is OCR'd from the blacked-out raster, where the covered
  text no longer exists).
"""

import os
import json
import sys
import glob
import shutil
import tempfile
import urllib.request

from coverup.textsearch import match_units, DEFAULT_PADDING
from coverup.i18n import get_system_locale

# Base language always made available; others are fetched on demand.
BASE_LANGUAGE = 'eng'

# Selectable model quality. We use the LSTM engine, so the two meaningful
# choices are the small integer models ("fast", the default — light downloads,
# great on clean text) and the float models ("best" — ~3-6x larger, more robust
# on poor scans). The legacy "tessdata" repo is intentionally not offered: it
# bundles the legacy engine we never use, so it is pure overhead.
TESSDATA_MODELS = {
    'fast': 'https://github.com/tesseract-ocr/tessdata_fast/raw/main/{code}.traineddata',
    'best': 'https://github.com/tesseract-ocr/tessdata_best/raw/main/{code}.traineddata',
}
DEFAULT_OCR_MODEL = 'fast'
_DOWNLOAD_TIMEOUT = 30

# All Tesseract language codes offered in the download dialog, mapped to
# their English display names.  Covers the full tessdata_fast/best catalogue
# of commonly-used languages; less-common scripts are intentionally omitted.
ALL_TESSERACT_LANGUAGE_NAMES = {
    'afr': 'Afrikaans',
    'ara': 'Arabic',
    'ben': 'Bengali',
    'bul': 'Bulgarian',
    'cat': 'Catalan',
    'ces': 'Czech',
    'chi_sim': 'Chinese (Simplified)',
    'chi_tra': 'Chinese (Traditional)',
    'cym': 'Welsh',
    'dan': 'Danish',
    'deu': 'German',
    'ell': 'Greek',
    'eng': 'English',
    'epo': 'Esperanto',
    'est': 'Estonian',
    'eus': 'Basque',
    'fas': 'Persian',
    'fin': 'Finnish',
    'fra': 'French',
    'gle': 'Irish',
    'glg': 'Galician',
    'guj': 'Gujarati',
    'heb': 'Hebrew',
    'hin': 'Hindi',
    'hrv': 'Croatian',
    'hun': 'Hungarian',
    'hye': 'Armenian',
    'ind': 'Indonesian',
    'isl': 'Icelandic',
    'ita': 'Italian',
    'jpn': 'Japanese',
    'kan': 'Kannada',
    'kat': 'Georgian',
    'kor': 'Korean',
    'lat': 'Latin',
    'lav': 'Latvian',
    'lit': 'Lithuanian',
    'mal': 'Malayalam',
    'mar': 'Marathi',
    'mkd': 'Macedonian',
    'mlt': 'Maltese',
    'nld': 'Dutch',
    'nor': 'Norwegian',
    'pol': 'Polish',
    'por': 'Portuguese',
    'ron': 'Romanian',
    'rus': 'Russian',
    'slk': 'Slovak',
    'slv': 'Slovenian',
    'spa': 'Spanish',
    'sqi': 'Albanian',
    'srp': 'Serbian',
    'swa': 'Swahili',
    'swe': 'Swedish',
    'tam': 'Tamil',
    'tel': 'Telugu',
    'tgl': 'Tagalog',
    'tha': 'Thai',
    'tur': 'Turkish',
    'ukr': 'Ukrainian',
    'urd': 'Urdu',
    'vie': 'Vietnamese',
}

# Map a CoverUP UI/system locale (2-letter) to a Tesseract language code.
# Covers every language CoverUP's interface supports, plus Arabic.
LOCALE_TO_TESSERACT = {
    'en': 'eng', 'de': 'deu', 'es': 'spa', 'fr': 'fra', 'it': 'ita',
    'pt': 'por', 'ro': 'ron', 'nl': 'nld', 'sv': 'swe', 'da': 'dan',
    'no': 'nor', 'is': 'isl', 'pl': 'pol', 'cs': 'ces', 'sk': 'slk',
    'bg': 'bul', 'sr': 'srp', 'hr': 'hrv', 'sl': 'slv', 'el': 'ell',
    'tr': 'tur', 'lt': 'lit', 'lv': 'lav', 'et': 'est', 'zh': 'chi_sim',
    'hi': 'hin', 'ar': 'ara',
}

# Cache the availability/version probe so we don't shell out on every call.
_AVAILABLE = None
_CONFIGURED = False
_OCR_MODEL = None


def _data_root():
    """Return CoverUP's user data directory (parent of config and tessdata)."""
    try:
        from appdirs import user_data_dir
        base = user_data_dir('CoverUP', 'digidigital')
    except Exception:
        base = os.path.join(os.path.expanduser('~'), '.coverup')
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        pass
    return base


def _config_path():
    return os.path.join(_data_root(), 'config.json')


def get_ocr_model():
    """Return the selected OCR model variant ('fast' or 'best'). Cached."""
    global _OCR_MODEL
    if _OCR_MODEL is not None:
        return _OCR_MODEL
    model = DEFAULT_OCR_MODEL
    try:
        with open(_config_path(), encoding='utf-8') as fh:
            model = json.load(fh).get('ocr_model', DEFAULT_OCR_MODEL)
    except Exception:
        pass
    _OCR_MODEL = model if model in TESSDATA_MODELS else DEFAULT_OCR_MODEL
    return _OCR_MODEL


def set_ocr_model(model):
    """Persist the OCR model variant choice. No-op for an unknown value."""
    global _OCR_MODEL
    if model not in TESSDATA_MODELS:
        return
    _OCR_MODEL = model
    cfg = {}
    try:
        with open(_config_path(), encoding='utf-8') as fh:
            cfg = json.load(fh)
    except Exception:
        cfg = {}
    cfg['ocr_model'] = model
    try:
        with open(_config_path(), 'w', encoding='utf-8') as fh:
            json.dump(cfg, fh)
    except Exception:
        pass


def _bundle_root():
    """Return the bundle directory holding shipped ``tesseract``/``tessdata``.

    Mirrors :func:`coverup.utils.get_resource_root`: in a PyInstaller build the
    data lives under ``_MEIPASS`` (== ``_internal`` for onedir) or next to the
    executable. Returns the first directory that contains a ``tesseract``
    subfolder, or ``None`` when running from source.
    """
    candidates = []
    if hasattr(sys, '_MEIPASS'):
        candidates.append(sys._MEIPASS)
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        candidates.append(os.path.join(exe_dir, '_internal'))
        candidates.append(exe_dir)
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, 'tesseract')):
            return c
    return None


def _bundled_tesseract_cmd():
    """Return the path to a bundled tesseract executable, or ``None``.

    On Linux this is a small wrapper script (it sets ``LD_LIBRARY_PATH`` to the
    bundled ``lib`` dir before exec'ing the real binary, so the app's own
    environment and its multiprocessing workers are never touched). On Windows
    it is ``tesseract.exe`` with its DLLs alongside.
    """
    root = _bundle_root()
    if not root:
        return None
    tdir = os.path.join(root, 'tesseract')
    for name in ('tesseract', 'tesseract.exe'):
        p = os.path.join(tdir, name)
        if os.path.isfile(p):
            return p
    return None


def _configure_tesseract():
    """Point pytesseract at the bundled binary when running from a build."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True
    cmd = _bundled_tesseract_cmd()
    if cmd:
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = cmd
        except Exception:
            pass


def is_available():
    """Return True if Tesseract OCR is usable.

    Prefers a tesseract bundled with the build (so installers work out of the
    box); otherwise falls back to one on PATH. The result is cached.
    """
    global _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    _AVAILABLE = False
    _configure_tesseract()
    if _bundled_tesseract_cmd() or shutil.which('tesseract'):
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            _AVAILABLE = True
        except Exception:
            _AVAILABLE = False
    return _AVAILABLE


def _tessdata_dir_for_model(model):
    """Return (creating if needed) tessdata dir for a specific model variant."""
    path = os.path.join(_data_root(), 'tessdata', model)
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
    return path


def user_tessdata_dir():
    """Return (creating if needed) CoverUP's writable tessdata dir.

    The directory is per-model (``tessdata/fast`` / ``tessdata/best``), so
    switching quality just points Tesseract at a different folder and each
    variant's models are downloaded and cached independently.
    """
    return _tessdata_dir_for_model(get_ocr_model())


def is_language_downloaded(code, model):
    """Return True if ``code``'s .traineddata file exists for the given model.

    Args:
        code: Tesseract language code (e.g. ``'spa'``).
        model: Model variant to check (``'fast'`` or ``'best'``).

    Returns:
        bool: True if the file is present in the model's tessdata directory.
    """
    path = os.path.join(_data_root(), 'tessdata', model, f'{code}.traineddata')
    return os.path.isfile(path)


def _source_tessdata_dirs():
    """Directories to look in when copying an already-present language file.

    Covers the PyInstaller bundle (so a shipped ``eng`` is reused, not
    re-downloaded) and the common system Tesseract locations across platforms.
    """
    dirs = []
    # PyInstaller bundle locations.
    if hasattr(sys, '_MEIPASS'):
        dirs.append(os.path.join(sys._MEIPASS, 'tessdata'))
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        dirs.append(os.path.join(exe_dir, 'tessdata'))
        dirs.append(os.path.join(exe_dir, '_internal', 'tessdata'))
    # Explicit override.
    if os.environ.get('TESSDATA_PREFIX'):
        dirs.append(os.environ['TESSDATA_PREFIX'])
    # System locations (Linux/macOS/Homebrew/Windows).
    dirs += glob.glob('/usr/share/tesseract-ocr/*/tessdata')
    dirs += ['/usr/share/tessdata', '/usr/local/share/tessdata',
             '/opt/homebrew/share/tessdata', '/opt/local/share/tessdata']
    if shutil.which('tesseract'):
        exe = shutil.which('tesseract')
        dirs.append(os.path.join(os.path.dirname(exe), 'tessdata'))
    seen, out = set(), []
    for d in dirs:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out


def _langs_in_dir(directory):
    """Return the set of language codes present in a tessdata directory."""
    if not directory or not os.path.isdir(directory):
        return set()
    return {os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(directory, '*.traineddata'))}


def _copy_from_sources(code, target, userdir):
    """Copy ``code``'s model from a system/bundle dir into target. Bool result."""
    for src in _source_tessdata_dirs():
        if os.path.realpath(src) == os.path.realpath(userdir):
            continue
        candidate = os.path.join(src, f'{code}.traineddata')
        if os.path.isfile(candidate):
            try:
                shutil.copyfile(candidate, target)
                return True
            except Exception:
                pass
    return False


def _download_model_streamed(code, target, userdir, model,
                             progress_cb=None, cancel_flag=None):
    """Download ``code``'s model for the given variant into target, atomically.

    Args:
        code: Tesseract language code.
        target: Destination ``.traineddata`` path.
        userdir: Directory used for the temp file.
        model: Model variant (``'fast'`` or ``'best'``).
        progress_cb: Optional ``callable(percent: int)`` called 0–100 during
            the download.  Only called when ``Content-Length`` is known.
        cancel_flag: Optional ``threading.Event``; if set, the download is
            abandoned and the temp file is removed.

    Returns:
        bool: True on success, False on failure or cancellation.
    """
    url = TESSDATA_MODELS[model].format(code=code)
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=userdir, suffix='.part')
        os.close(fd)
        with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT) as resp:
            total = int(resp.headers.get('Content-Length') or 0)
            downloaded = 0
            with open(tmp, 'wb') as fh:
                while True:
                    if cancel_flag is not None and cancel_flag.is_set():
                        return False
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb and total > 0:
                        progress_cb(min(99, int(downloaded * 100 / total)))
        if progress_cb:
            progress_cb(100)
        os.replace(tmp, target)  # atomic
        return True
    except Exception:
        return False
    finally:
        try:
            if tmp and os.path.isfile(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _download_model(code, target, userdir):
    """Download ``code``'s model for the active variant into target, atomically."""
    return _download_model_streamed(code, target, userdir, get_ocr_model())


def ensure_language(code):
    """Make a single language available in the active model's tessdata dir.

    For the ``fast`` variant, a model already on the machine (bundle/system) is
    reused before downloading. For ``best``, the high-accuracy model is
    downloaded first — copying the system model (which is fast-grade) would
    defeat the choice — with the local copy kept only as an offline fallback.
    Downloads are atomic, so a partial transfer never leaves a corrupt model.

    Args:
        code: A Tesseract language code (e.g. ``'spa'``, ``'chi_sim'``).

    Returns:
        bool: True if the language is now available locally, False otherwise
        (e.g. download failed with no network).
    """
    if not code:
        return False
    userdir = user_tessdata_dir()
    target = os.path.join(userdir, f'{code}.traineddata')
    if os.path.isfile(target):
        return True

    if get_ocr_model() == 'best':
        return _download_model(code, target, userdir) or _copy_from_sources(code, target, userdir)
    return _copy_from_sources(code, target, userdir) or _download_model(code, target, userdir)


def ensure_languages(lang):
    """Ensure every code in a ``+``-joined language string is available.

    Args:
        lang: A Tesseract language string, e.g. ``'spa+eng'``.

    Returns:
        str: A language string containing only the codes that are now
        available (so OCR runs with whatever resolved), or ``''`` if none did.
    """
    codes = [c for c in (lang or '').split('+') if c]
    ok = [c for c in codes if ensure_language(c)]
    return '+'.join(ok)


def ensure_language_for_model(code, model, progress_cb=None, cancel_flag=None):
    """Make a single language available for a specific model variant.

    Like :func:`ensure_language` but operates on an explicit *model* without
    touching the global ``_OCR_MODEL``.  Safe to call from a background thread.

    Args:
        code: A Tesseract language code (e.g. ``'spa'``).
        model: Model variant to use (``'fast'`` or ``'best'``).
        progress_cb: Optional ``callable(percent: int)`` for download progress.
        cancel_flag: Optional ``threading.Event``; download aborts if set.

    Returns:
        bool: True if the language is now available locally.
    """
    if not code or model not in TESSDATA_MODELS:
        return False
    userdir = _tessdata_dir_for_model(model)
    target = os.path.join(userdir, f'{code}.traineddata')
    if os.path.isfile(target):
        if progress_cb:
            progress_cb(100)
        return True
    if model == 'best':
        return (_download_model_streamed(code, target, userdir, model,
                                         progress_cb, cancel_flag)
                or _copy_from_sources(code, target, userdir))
    return (_copy_from_sources(code, target, userdir)
            or _download_model_streamed(code, target, userdir, model,
                                         progress_cb, cancel_flag))


def available_languages_for_model(model):
    """Return sorted language codes already downloaded for a specific model."""
    return sorted(_langs_in_dir(_tessdata_dir_for_model(model)) - {'osd'})


def system_language():
    """Return the Tesseract code matching the system/UI locale (default eng)."""
    return LOCALE_TO_TESSERACT.get(get_system_locale(), BASE_LANGUAGE)


def available_languages():
    """Return language codes offered in the UI.

    The union of everything already present locally (CoverUP's dir, the bundle,
    and the system) plus the base language and the system-locale language —
    the latter two may not be downloaded yet, but :func:`ensure_languages`
    fetches them on first use, so they are safe to offer. ``osd`` (the
    orientation model, not a language) is excluded.
    """
    if not is_available():
        return []
    langs = set()
    langs |= _langs_in_dir(user_tessdata_dir())
    for d in _source_tessdata_dirs():
        langs |= _langs_in_dir(d)
    langs.add(BASE_LANGUAGE)
    langs.add(system_language())
    langs.discard('osd')
    return sorted(langs)


def default_language():
    """Best default OCR language string: system-locale language + English.

    Both are guaranteed available after :func:`ensure_languages` runs, so this
    can confidently propose e.g. ``'spa+eng'`` even before the Spanish model
    has been downloaded.
    """
    sys_lang = system_language()
    if sys_lang == BASE_LANGUAGE:
        return BASE_LANGUAGE
    return f'{sys_lang}+{BASE_LANGUAGE}'


def _tessdata_config():
    """Config string pinning Tesseract to CoverUP's self-sufficient dir."""
    return f'--tessdata-dir {user_tessdata_dir()}'


def ocr_words(pil_image, lang=None, min_conf=30):
    """Run OCR on an image and return recognised words with pixel boxes.

    Missing language models are fetched on demand before OCR runs.

    Args:
        pil_image: A PIL image (already redacted, for the export use case).
        lang: Tesseract language string (e.g. ``'spa+eng'``). Defaults to
              :func:`default_language`.
        min_conf: Drop words whose OCR confidence is below this (0-100).

    Returns:
        list[dict]: One dict per word with keys ``text``, ``left``, ``top``,
        ``width``, ``height``, ``conf``, ``block``, ``par``, ``line``,
        ``word`` — coordinates in image pixels (top-left origin). Empty if OCR
        is unavailable or fails.
    """
    if not is_available():
        return []
    import pytesseract
    from pytesseract import Output
    from PIL import Image, ImageOps

    if lang is None:
        lang = default_language()
    lang = ensure_languages(lang) or BASE_LANGUAGE

    # Preprocess for accuracy: Tesseract works best on high-contrast, ~300 DPI
    # input. CoverUP rasterises pages at ~144 DPI, which is fine to display but
    # low for OCR, so upscale small images and flatten to high-contrast grey.
    # Boxes are reported back in the *original* pixel space (divided by the
    # upscale factor), so callers' coordinate maths is unaffected.
    factor = 1
    try:
        short_side = min(pil_image.size)
        if short_side:
            factor = max(1, min(3, round(2200 / short_side)))
        work = ImageOps.autocontrast(pil_image.convert('L'))
        if factor > 1:
            work = work.resize((work.width * factor, work.height * factor),
                               resample=Image.Resampling.LANCZOS)
    except Exception:
        work, factor = pil_image, 1

    try:
        data = pytesseract.image_to_data(
            work, lang=lang, config=_tessdata_config(), output_type=Output.DICT
        )
    except Exception:
        return []
    finally:
        if work is not pil_image:
            try:
                work.close()
            except Exception:
                pass

    words = []
    for i in range(len(data['text'])):
        text = data['text'][i]
        if not text or not text.strip():
            continue
        try:
            conf = float(data['conf'][i])
        except (TypeError, ValueError):
            conf = -1
        if conf < min_conf:
            continue
        words.append({
            'text': text,
            'left': int(int(data['left'][i]) / factor),
            'top': int(int(data['top'][i]) / factor),
            'width': int(int(data['width'][i]) / factor),
            'height': int(int(data['height'][i]) / factor),
            'conf': conf,
            'block': int(data['block_num'][i]),
            'par': int(data['par_num'][i]),
            'line': int(data['line_num'][i]),
            'word': int(data['word_num'][i]),
        })
    return words


def _words_to_units(words):
    """Turn OCR word boxes into character units for :func:`match_units`.

    Each word box is split into per-character sub-boxes by even width division
    (OCR gives only word-level geometry), so multi-token patterns like phone
    numbers can still be matched and barred line by line. A synthetic space is
    inserted between words on the same line and a newline between lines, so the
    reconstructed text reads naturally.

    Args:
        words: Output of :func:`ocr_words`, in document order.

    Returns:
        list[tuple]: ``(char, box_or_None)`` units in image pixels.
    """
    units = []
    prev_key = None
    for w in words:
        key = (w['block'], w['par'], w['line'])
        if prev_key is not None:
            if key == prev_key:
                units.append((' ', None))
            else:
                units.append(('\n', None))
        prev_key = key

        text = w['text']
        n = max(len(text), 1)
        char_w = w['width'] / n
        for j, c in enumerate(text):
            left = w['left'] + j * char_w
            box = (left, w['top'], left + char_w, w['top'] + w['height'])
            units.append((c, box))
    return units


def find_boxes_ocr(pil_image, patterns=None, keywords=None, lang=None,
                   padding=DEFAULT_PADDING, min_conf=30):
    """Detect sensitive text on a page image via OCR and return redaction boxes.

    The OCR counterpart of :func:`coverup.textsearch.find_boxes_digital`, for
    scanned PDFs and imported images that have no digital text layer.

    Args:
        pil_image: The page image to scan.
        patterns: Iterable of keys from :data:`coverup.textsearch.PATTERN_DEFS`.
        keywords: Iterable of literal keyword strings.
        lang: Tesseract language string; defaults to :func:`default_language`.
        padding: Pixels to grow each rectangle on every side.
        min_conf: Minimum OCR confidence for a word to be considered.

    Returns:
        list[tuple]: ``(start_xy, end_xy)`` rectangles in image pixel
        coordinates. Empty if OCR is unavailable or nothing matched.
    """
    words = ocr_words(pil_image, lang=lang, min_conf=min_conf)
    if not words:
        return []
    units = _words_to_units(words)
    return match_units(units, patterns, keywords, padding, ocr_tolerant=True)
