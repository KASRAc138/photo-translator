"""Local web UI."""

from __future__ import annotations

import json
import logging
import mimetypes
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import imageio
from .config import Config, resource_dir
from .geometry import TextBox

log = logging.getLogger("pt.webui")

STATIC = resource_dir("static") or (Path(__file__).resolve().parent / "static")

# Offered in the language picker. Argos publishes ~100 packs but almost all
# route through English; these are the ones worth surfacing by default.
LANGUAGES = [
    ("en", "English"), ("fa", "Persian / فارسی"), ("de", "German"),
    ("fr", "French"), ("es", "Spanish"), ("it", "Italian"),
    ("ar", "Arabic"), ("tr", "Turkish"), ("ru", "Russian"),
    ("zh", "Chinese"), ("ja", "Japanese"), ("nl", "Dutch"),
    ("pt", "Portuguese"), ("pl", "Polish"), ("sv", "Swedish"),
]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@dataclass
class PageState:
    """Everything known about one image, cached between requests."""

    path: Path
    boxes: list[TextBox] = field(default_factory=list)
    source_lang: str = "auto"
    detected_lang: str | None = None
    rendered: bytes | None = None
    error: str | None = None
    analysed: bool = False


class Session:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pages: dict[str, PageState] = {}
        self.lock = threading.Lock()
        self.progress = {"active": False, "done": 0, "total": 0, "current": "", "message": ""}
        self._pipeline = None

    # Engines are built lazily and shared: RapidOCR takes several seconds to
    # construct and would otherwise be rebuilt per image.
    def pipeline(self):
        if self._pipeline is None:
            from .pipeline import Pipeline
            self._pipeline = Pipeline(self.cfg)
        return self._pipeline

    def set_progress(self, **kwargs) -> None:
        self.progress.update(kwargs)


# ---------------------------------------------------------------------------
# Work
# ---------------------------------------------------------------------------


def analyse(session: Session, state: PageState) -> PageState:
    """OCR + orientation + translation for one page. Does not render."""
    from .translate import ArgosTranslator

    cfg = session.cfg
    pipeline = session.pipeline()
    image = imageio.load(state.path)

    boxes = pipeline.ocr.detect(image)
    state.boxes = boxes
    if not boxes:
        state.analysed = True
        return state

    source = cfg.source_lang
    if source == "auto":
        source = ArgosTranslator.detect_language(" ".join(b.text for b in boxes[:12]))
    state.detected_lang = source
    state.source_lang = source

    translations = pipeline.translator.translate([b.text for b in boxes], source, cfg.target_lang)
    for box, translated in zip(boxes, translations):
        box.translated = translated
    state.analysed = True
    return state


def render_page(session: Session, state: PageState) -> bytes:
    """Render the current boxes to PNG bytes, honouring per-box enable flags."""
    import io

    from .render import render

    image = imageio.load(state.path)
    active = [b for b in state.boxes if not b.meta.get("disabled")]
    canvas = render(image, active, session.cfg)
    buffer = io.BytesIO()
    canvas.save(buffer, "PNG", optimize=False)
    state.rendered = buffer.getvalue()
    return state.rendered


def box_json(box: TextBox, index: int) -> dict:
    return {
        "i": index,
        "quad": [[round(x, 1), round(y, 1)] for x, y in box.quad],
        "text": box.text,
        "translated": box.translated or "",
        "orientation": box.orientation,
        "confidence": round(box.confidence, 3),
        "vertical": box.is_vertical,
        "disabled": bool(box.meta.get("disabled")),
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    session: Session = None  # type: ignore[assignment]
    server_version = "PhotoTranslator"

    def log_message(self, fmt, *args):  # quieter than the default
        log.debug(fmt, *args)

    # -- helpers ---------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser navigated away mid-response

    def _json(self, payload, status: int = 200):
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _payload(self) -> dict:
        try:
            return json.loads(self._body() or b"{}")
        except json.JSONDecodeError:
            return {}

    def _state(self, key: str) -> PageState | None:
        return self.session.pages.get(key)

    # -- routing ---------------------------------------------------------

    def do_GET(self):
        route = urlparse(self.path)
        path, query = route.path, parse_qs(route.query)

        if path in ("/", "/index.html"):
            return self._static("index.html")
        if path.startswith("/static/"):
            return self._static(path[len("/static/"):])
        if path == "/api/config":
            cfg = self.session.cfg
            return self._json({
                "languages": [{"code": c, "name": n} for c, n in LANGUAGES],
                "target": cfg.target_lang,
                "source": cfg.source_lang,
                "erase": cfg.erase_mode,
                "input_dir": str(cfg.input_dir),
                "output_dir": str(cfg.output_dir),
            })
        if path == "/api/files":
            return self._json({"files": self._list_files()})
        if path == "/api/progress":
            return self._json(self.session.progress)
        if path == "/api/boxes":
            state = self._state(query.get("file", [""])[0])
            if not state:
                return self._json({"error": "unknown file"}, 404)
            return self._json({
                "analysed": state.analysed,
                "source": state.detected_lang,
                "error": state.error,
                "boxes": [box_json(b, i) for i, b in enumerate(state.boxes)],
            })
        if path == "/api/image":
            return self._image(query)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        route = urlparse(self.path)
        path = route.path

        if path == "/api/upload":
            return self._upload()
        if path == "/api/settings":
            data = self._payload()
            cfg = self.session.cfg
            if "target" in data:
                cfg.target_lang = str(data["target"])
            if "source" in data:
                cfg.source_lang = str(data["source"])
            if "erase" in data:
                cfg.erase_mode = str(data["erase"])
            # Language or erase changes invalidate every rendered preview.
            for state in self.session.pages.values():
                state.rendered = None
                if "target" in data or "source" in data:
                    state.analysed = False
            return self._json({"ok": True})
        if path == "/api/run":
            return self._run(self._payload())
        if path == "/api/edit":
            return self._edit(self._payload())
        if path == "/api/save":
            return self._save(self._payload())
        return self._json({"error": "not found"}, 404)

    # -- endpoints -------------------------------------------------------

    def _static(self, name: str):
        target = (STATIC / name).resolve()
        try:
            target.relative_to(STATIC.resolve())
        except ValueError:
            return self._json({"error": "forbidden"}, 403)
        if not target.is_file():
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send(200, target.read_bytes(), ctype)

    def _list_files(self) -> list[dict]:
        cfg = self.session.cfg
        cfg.ensure_folders()
        out = []
        for path in imageio.iter_images(cfg.input_dir):
            key = path.name
            state = self.session.pages.get(key)
            if state is None:
                state = PageState(path=path)
                self.session.pages[key] = state
            out.append({
                "name": key,
                "analysed": state.analysed,
                "boxes": len(state.boxes),
                "rendered": state.rendered is not None,
                "error": state.error,
            })
        return out

    def _upload(self):
        name = unquote(self.headers.get("X-Filename", "") or "")
        data = self._body()
        if not name or not data:
            return self._json({"error": "missing filename or body"}, 400)
        # Flatten any path the browser sent; only the basename is ever used.
        safe = Path(name).name
        if not imageio.is_supported(safe):
            return self._json({"error": f"unsupported file type: {safe}"}, 400)
        cfg = self.session.cfg
        cfg.ensure_folders()
        target = cfg.input_dir / safe
        target.write_bytes(data)
        self.session.pages.pop(safe, None)
        return self._json({"ok": True, "name": safe})

    def _image(self, query: dict):
        state = self._state(query.get("file", [""])[0])
        if not state:
            return self._json({"error": "unknown file"}, 404)
        kind = query.get("kind", ["original"])[0]
        if kind == "result":
            if state.rendered is None:
                return self._json({"error": "not rendered"}, 404)
            return self._send(200, state.rendered, "image/png")
        # Originals are re-encoded through the canonical load path so the
        # browser sees exactly the upright pixels the pipeline works on --
        # otherwise a rotated JPEG would preview one way and process another.
        import io
        buffer = io.BytesIO()
        imageio.load(state.path).pil.save(buffer, "JPEG", quality=88)
        self._send(200, buffer.getvalue(), "image/jpeg")

    def _run(self, data: dict):
        names = data.get("files") or [f["name"] for f in self._list_files()]
        if self.session.progress.get("active"):
            return self._json({"error": "already running"}, 409)

        def worker():
            session = self.session
            session.set_progress(active=True, done=0, total=len(names), current="", message="")
            try:
                for i, name in enumerate(names):
                    state = session.pages.get(name)
                    if state is None:
                        continue
                    session.set_progress(current=name, done=i, message="reading text")
                    with session.lock:
                        try:
                            state.error = None
                            if not state.analysed:
                                analyse(session, state)
                            session.set_progress(message="typesetting")
                            render_page(session, state)
                        except Exception as exc:      # one bad page must not stop the batch
                            log.exception("failed on %s", name)
                            state.error = str(exc)
                    session.set_progress(done=i + 1)
            finally:
                session.set_progress(active=False, current="", message="")

        threading.Thread(target=worker, daemon=True).start()
        return self._json({"ok": True, "total": len(names)})

    def _edit(self, data: dict):
        """Apply a per-box correction and re-render. Skips OCR and translation."""
        state = self._state(data.get("file", ""))
        if not state:
            return self._json({"error": "unknown file"}, 404)
        index = int(data.get("index", -1))
        if not 0 <= index < len(state.boxes):
            return self._json({"error": "bad box index"}, 400)

        box = state.boxes[index]
        if "translated" in data:
            box.translated = str(data["translated"])
        if "disabled" in data:
            box.meta["disabled"] = bool(data["disabled"])
        if "orientation" in data:
            box.orientation = int(data["orientation"]) % 360

        with self.session.lock:
            try:
                render_page(self.session, state)
            except Exception as exc:
                return self._json({"error": str(exc)}, 500)
        return self._json({"ok": True, "box": box_json(box, index)})

    def _save(self, data: dict):
        names = data.get("files") or list(self.session.pages)
        cfg = self.session.cfg
        cfg.ensure_folders()
        written = []
        for name in names:
            state = self.session.pages.get(name)
            if not state or state.rendered is None:
                continue
            stem = Path(name).stem
            target = cfg.output_dir / f"{stem}_{cfg.target_lang}.png"
            target.write_bytes(state.rendered)
            written.append(target.name)
        return self._json({"ok": True, "written": written, "folder": str(cfg.output_dir)})


# ---------------------------------------------------------------------------


def serve(cfg: Config | None = None, port: int = 8731, open_browser: bool = True) -> None:
    cfg = cfg or Config()
    cfg.ensure_folders()

    Handler.session = Session(cfg)
    # Port 0 would be simpler but the URL must be predictable enough to retype.
    for attempt in range(20):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", port + attempt), Handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"Could not bind a port in {port}-{port + 19}.")

    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"\n  Photo Translator is running at  {url}")
    print(f"  input   {cfg.input_dir}")
    print(f"  output  {cfg.output_dir}")
    print("\n  Close this window to stop.\n")

    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        httpd.server_close()
