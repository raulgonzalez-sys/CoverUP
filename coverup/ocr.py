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
import sys
import glob
import shutil
import tempfile
import urllib.request

from coverup.textsearch import match_units, DEFAULT_PADDING
from coverup.i18n import get_system_locale

# Base language always made available; others are fetched on demand.
BASE_LANGUAGE = 'eng'

# Where single language files are fetched from when missing locally. The
# ``fast`` models are ~1/3 the size of the standard ones with negligible
# accuracy loss for this use case.
TESSDATA_BASE_URL = 'https://github.com/tesseract-ocr/tessdata_fast/raw/main/{code}.traineddata'
_DOWNLOAD_TIMEOUT = 30

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
_USER_TESSDATA = None
_CONFIGURED = False


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


def user_tessdata_dir():
    """Return (creating if needed) CoverUP's own writable tessdata directory."""
    global _USER_TESSDATA
    if _USER_TESSDATA is not None:
        return _USER_TESSDATA
    try:
        from appdirs import user_data_dir
        base = user_data_dir('CoverUP', 'digidigital')
    except Exception:
        base = os.path.join(os.path.expanduser('~'), '.coverup')
    path = os.path.join(base, 'tessdata')
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
    _USER_TESSDATA = path
    return path


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


def ensure_language(code):
    """Make a single language available in CoverUP's tessdata directory.

    Resolution order: already present → copy from a system/bundle directory →
    download from ``tessdata_fast``. Downloads are written atomically so a
    partial transfer never leaves a corrupt model behind.

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

    # Reuse a copy already on the machine (bundle or system) before downloading.
    for src in _source_tessdata_dirs():
        if os.path.realpath(src) == os.path.realpath(userdir):
            continue
        candidate = os.path.join(src, f'{code}.traineddata')
        if os.path.isfile(candidate):
            try:
                shutil.copyfile(candidate, target)
                return True
            except Exception:
                pass  # fall through to download

    # Download the single model file.
    url = TESSDATA_BASE_URL.format(code=code)
    try:
        fd, tmp = tempfile.mkstemp(dir=userdir, suffix='.part')
        os.close(fd)
        with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT) as resp:
            data = resp.read()
        if not data:
            os.remove(tmp)
            return False
        with open(tmp, 'wb') as fh:
            fh.write(data)
        os.replace(tmp, target)  # atomic
        return True
    except Exception:
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False


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

    if lang is None:
        lang = default_language()
    lang = ensure_languages(lang) or BASE_LANGUAGE

    try:
        data = pytesseract.image_to_data(
            pil_image, lang=lang, config=_tessdata_config(), output_type=Output.DICT
        )
    except Exception:
        return []

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
            'left': int(data['left'][i]),
            'top': int(data['top'][i]),
            'width': int(data['width'][i]),
            'height': int(data['height'][i]),
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
    return match_units(units, patterns, keywords, padding)
