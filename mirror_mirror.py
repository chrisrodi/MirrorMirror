# mirror_mirror.py
# Captures exactly the glass region using Qt grabWindow (logical coords).
# Glass pane blocks clicks, can be dragged from center or border, and resized via edges/corners.
# Pauses when adjusting/moving/resizing or when OCR text hasn't changed.
# Overlay draws no text to avoid OCR feedback loops. Green answer is shown in the header.
#
# deps: pyqt6 pillow pytesseract openai python-dotenv
# version b0.0.0
import os, sys, time, threading, shutil
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv
dotenv_path = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=dotenv_path)

from PyQt6.QtCore import Qt, QRect, QPoint, QTimer, pyqtSignal, QObject, QSize, pyqtSlot
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QGuiApplication, QImage, QRegion
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QFrame, QVBoxLayout, QHBoxLayout, QPushButton,
    QMessageBox, QSpacerItem, QSizePolicy, QGraphicsDropShadowEffect, QColorDialog
)

# ---------- Tesseract path
import pytesseract
from PIL import Image, ImageOps, ImageFilter
if os.path.exists("/opt/homebrew/bin/tesseract"):
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
elif os.path.exists("/usr/local/bin/tesseract"):
    pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

_TESS = shutil.which("tesseract") or pytesseract.pytesseract.tesseract_cmd
pytesseract.pytesseract.tesseract_cmd = _TESS

# ---------- OpenAI
from openai import OpenAI
client = OpenAI()
MODEL = os.getenv("READNWRITE_MODEL", "gpt-4o")

DEFAULT_PANE_COLOR = QColor(212, 175, 55, 230)
PANE_COLOR = QColor(DEFAULT_PANE_COLOR)

# ================== helpers ==================
@dataclass
class CaptureRect:
    left:int; top:int; width:int; height:int

def pil_from_qimage(qimg: QImage) -> Image.Image:
    qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    width = qimg.width()
    height = qimg.height()
    ptr = qimg.bits()
    ptr.setsize(qimg.sizeInBytes())
    arr = bytes(ptr)
    return Image.frombuffer("RGBA", (width, height), arr, "raw", "RGBA", 0, 1).convert("RGB")


# ================== transparent glass pane with perimeter drag + grips ==================
class GlassPane(QWidget):
    """Gold-framed, draggable/resizable overlay.
       - Drag from center or any border to move
       - Single-click near center to trigger an immediate OCR capture
       - Resize via large corners + edge strips
       - Blocks clicks (no click-through)
       - Center is painted with alpha=1 so macOS hit-tests it
    """
    EDGE_T = 12
    CORNER = 26
    BAR_H  = 28
    MIN_W  = 1
    MIN_H  = 1
    HIT_ALPHA = 1  # barely visible fill so the center is hit-testable
    OCR_CLICK_RADIUS = 40  # px radius around center that counts as "click" for OCR

    adjusting = pyqtSignal(bool)
    geometry_changed = pyqtSignal(QRect)
    pauseRequested = pyqtSignal(bool)  # True=start drag -> pause, False=end drag -> resume (debounced by header)
    ocrRequested = pyqtSignal()

    def __init__(self, geo: QRect, corner_radius: int = 22):
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)  # not click-through
        self.setStyleSheet("background: transparent;")
        self.setGeometry(geo)
        self._corner_radius = corner_radius

        # labels (let clicks pass to pane so center-drag works)
        self.black = QLabel(self); self.green = QLabel(self)
        for L in (self.black, self.green):
            L.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            L.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            L.setStyleSheet("background: transparent;")
            L.setWordWrap(True)
        bf = QFont(); bf.setPointSize(22); bf.setBold(True)
        gf = QFont(); gf.setPointSize(22); gf.setBold(True)
        self.black.setFont(bf); self.black.setStyleSheet("color:#222;")
        self.green.setFont(gf); self.green.setStyleSheet("color:#1aa34a;")

        # drag bookkeeping
        self._drag_origin = None
        self._press_global = QPoint()
        self._press_pos = QPoint()
        self._dragging = False  # track if a move actually happened

        # grip state
        self._grip_active = None
        self._grip_origin_geo = None
        self._grip_press_global = QPoint()

        # robust adjusting flag
        self._is_adjusting = False

        # build hit areas
        self._make_perimeter_bands()
        self._make_resize_grips()
        self._reflow()

        # ensure the whole rect is hit-testable
        self.setMask(QRegion(self.rect()))
        self.set_texts("", "")  # keep overlay text blank

        # glow pulse
        self._glow_level = 0.0  # 0..1 fade
        self._glow_timer = QTimer(self)
        self._glow_timer.setInterval(16)  # ~60 FPS
        self._glow_timer.timeout.connect(self._advance_glow)
        self._glow_decay_ms = 650  # total glow duration (longer so it's visible)
        self._glow_started_at = 0.0

    # ----- glow API -----
    def trigger_glow(self):
        self._glow_started_at = time.time()
        self._glow_level = 1.0
        if not self._glow_timer.isActive():
            self._glow_timer.start()
        self.update()

    def _advance_glow(self):
        elapsed = (time.time() - self._glow_started_at) * 1000.0
        t = max(0.0, 1.0 - (elapsed / self._glow_decay_ms))  # linear fade 1→0
        self._glow_level = t
        if t <= 0.0:
            self._glow_timer.stop()
            self._glow_level = 0.0
        self.update()

    # ----- adjusting helpers -----
    def _begin_adjusting(self):
        if not self._is_adjusting:
            self._is_adjusting = True
            self.adjusting.emit(True)

    def _end_adjusting(self):
        if self._is_adjusting:
            self._is_adjusting = False
            self.adjusting.emit(False)

    # ---------- center-drag the whole pane ----------
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = self.geometry()
            self._press_global = e.globalPosition().toPoint()
            self._press_pos = e.position().toPoint()
            self._dragging = False  # reset; decide on click vs drag on move

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton and self._drag_origin is not None:
            if not self._dragging:
                self._dragging = True
                self._begin_adjusting()
                self.pauseRequested.emit(True)  # << pause now # ensure we pause even if press was missed
            delta = e.globalPosition().toPoint() - self._press_global
            self.setGeometry(self._drag_origin.translated(delta))
            self.geometry_changed.emit(self.geometry())

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            if self._dragging:
                self._drag_origin = None
                self._dragging = False
                self._end_adjusting()
                self.geometry_changed.emit(self.geometry())
                self.pauseRequested.emit(False)  # << allow resume (header debounces)
            else:
                self._drag_origin = None
                center = self.rect().center()
                if (self._press_pos - center).manhattanLength() <= self.OCR_CLICK_RADIUS:
                    self.ocrRequested.emit()
            self._press_pos = QPoint()

    # ---------- perimeter move bands ----------
    def _make_perimeter_bands(self):
        self.move_bands = {}
        for side in ("top", "bottom", "left", "right"):
            band = QWidget(self)
            band.setObjectName(f"move_{side}")
            band.setCursor(Qt.CursorShape.SizeAllCursor)
            band.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
            band.setStyleSheet(
                f"background: rgba({PANE_COLOR.red()},{PANE_COLOR.green()},{PANE_COLOR.blue()},0.12);"
            )
            band.mousePressEvent = self._band_press
            band.mouseMoveEvent  = self._band_move
            band.mouseReleaseEvent = self._band_release
            self.move_bands[side] = band

    def _band_press(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = self.geometry()
            self._press_global = e.globalPosition().toPoint()
            self._begin_adjusting()
            self.pauseRequested.emit(True)  # << pause now

    def _band_move(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton and self._drag_origin is not None:
            self._begin_adjusting()
            delta = e.globalPosition().toPoint() - self._press_global
            self.setGeometry(self._drag_origin.translated(delta))
            self.geometry_changed.emit(self.geometry())

    def _band_release(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = None
            self._end_adjusting()
            self.geometry_changed.emit(self.geometry())
            self.pauseRequested.emit(False)  # << allow resume

    # ---------- resize grips ----------
    def _make_resize_grips(self):
        self.grips = {}
        def mk(kind: str, cursor: Qt.CursorShape):
            g = QWidget(self)
            g.setObjectName(f"grip_{kind}")
            g.setCursor(cursor)
            g.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
            g.setStyleSheet(
                f"""
                QWidget#grip_nw, QWidget#grip_ne, QWidget#grip_sw, QWidget#grip_se {{
                    background: transparent;
                }}
                QWidget#grip_n:hover, QWidget#grip_s:hover, QWidget#grip_w:hover, QWidget#grip_e:hover,
                 QWidget#grip_nw:hover, QWidget#grip_ne:hover, QWidget#grip_sw:hover, QWidget#grip_se:hover {{
                    background: rgba({PANE_COLOR.red()},{PANE_COLOR.green()},{PANE_COLOR.blue()},0.20);
                    border-radius: 6px;
               }}
            """
            )
            g.mousePressEvent = lambda e, k=kind: self._grip_press(e, k)
            g.mouseMoveEvent  = lambda e, k=kind: self._grip_move(e, k)
            g.mouseReleaseEvent = lambda e, k=kind: self._grip_release(e, k)
            self.grips[kind] = g

        mk("nw", Qt.CursorShape.SizeFDiagCursor)
        mk("ne", Qt.CursorShape.SizeBDiagCursor)
        mk("sw", Qt.CursorShape.SizeBDiagCursor)
        mk("se", Qt.CursorShape.SizeFDiagCursor)
        mk("n",  Qt.CursorShape.SizeVerCursor)
        mk("s",  Qt.CursorShape.SizeVerCursor)
        mk("w",  Qt.CursorShape.SizeHorCursor)
        mk("e",  Qt.CursorShape.SizeHorCursor)

    def _grip_press(self, e, kind):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        self._grip_active = kind
        self._grip_origin_geo = self.geometry()
        self._grip_press_global = e.globalPosition().toPoint()
        self._begin_adjusting()
        self.pauseRequested.emit(True)  # << pause now

    def _grip_move(self, e, kind):
        if (e.buttons() & Qt.MouseButton.LeftButton) == 0 or self._grip_active != kind:
            return
        self._begin_adjusting()

        delta = e.globalPosition().toPoint() - self._grip_press_global
        x = self._grip_origin_geo.x(); y = self._grip_origin_geo.y()
        w = self._grip_origin_geo.width(); h = self._grip_origin_geo.height()
        dx, dy = delta.x(), delta.y()

        if "w" in kind:
            nx, nw = x + dx, w - dx
            if nw >= self.MIN_W: x, w = nx, nw
        if "e" in kind:
            nw = w + dx
            if nw >= self.MIN_W: w = nw
        if "n" in kind:
            ny, nh = y + dy, h - dy
            if nh >= self.MIN_H: y, h = ny, nh
        if "s" in kind:
            nh = h + dy
            if nh >= self.MIN_H: h = nh

        self.setGeometry(QRect(x, y, w, h))
        self.geometry_changed.emit(self.geometry())

    def _grip_release(self, e, kind):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        self._grip_active = None
        self._grip_origin_geo = None
        self._end_adjusting()
        self.geometry_changed.emit(self.geometry())
        self.pauseRequested.emit(False)  # << allow resume

    # ---------- layout ----------
    def _layout_perimeter_bands(self):
        r = self.rect(); t = self.EDGE_T; g = self.CORNER
        # inset bands so they don't overlap corner grips
        self.move_bands["top"].setGeometry(g, 0, max(0, r.width() - 2*g), t)
        self.move_bands["bottom"].setGeometry(g, r.height()-t, max(0, r.width() - 2*g), t)
        self.move_bands["left"].setGeometry(0, g, t, max(0, r.height() - 2*g))
        self.move_bands["right"].setGeometry(r.width()-t, g, t, max(0, r.height() - 2*g))

    def _layout_grips(self):
        r = self.rect(); g = self.CORNER; t = self.EDGE_T
        self.grips["nw"].setGeometry(0, 0, g, g)
        self.grips["ne"].setGeometry(r.width()-g, 0, g, g)
        self.grips["sw"].setGeometry(0, r.height()-g, g, g)
        self.grips["se"].setGeometry(r.width()-g, r.height()-g, g, g)
        self.grips["n"].setGeometry(g, 0, r.width()-2*g, t)
        self.grips["s"].setGeometry(g, r.height()-t, r.width()-2*g, t)
        self.grips["w"].setGeometry(0, g, t, r.height()-2*g)
        self.grips["e"].setGeometry(r.width()-t, g, t, r.height()-2*g)
        for k in ("nw", "ne", "sw", "se"):
            self.grips[k].raise_()
        for k in ("n", "s", "w", "e"):
            self.grips[k].raise_()

    def _reflow(self):
        m = 14
        w = self.width() - 2*m
        # (text labels kept blank, but kept for future use)
        self.black.setGeometry(m, m + self.BAR_H//2, w, int(self.height()*0.35))
        self.green.setGeometry(m, self.black.y() + self.black.height() + 6, w, int(self.height()*0.55))
        self._layout_perimeter_bands()
        self._layout_grips()

    def resizeEvent(self, _):
        self._reflow()
        self.setMask(QRegion(self.rect()))

    # ---------- paint ----------
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # fill center with 1-alpha for hit-testing
        p.fillRect(self.rect(), QColor(0, 0, 0, self.HIT_ALPHA))

        r = self.rect().adjusted(2, 2, -2, -2)

        # --- yellowish-white pulse when active ---
        # --- yellowish-white pulse when active (additive so it "blooms") ---
        if self._glow_level > 0.0:
            glow_width = 14 + int(14 * self._glow_level)
            glow_alpha = int(220 * self._glow_level)
            pulse_color = QColor(255, 248, 210, glow_alpha)
            p.save()
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
            pen = QPen(pulse_color, glow_width,
                       cap=Qt.PenCapStyle.RoundCap,
                       join=Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r, self._corner_radius, self._corner_radius)
            p.restore()

        # --- normal gold border ---
        p.setPen(QPen(PANE_COLOR, 4))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(r, self._corner_radius, self._corner_radius)

    def set_texts(self, black: str, green: str):
        # Keep overlay text blank (answer is in header)
        self.black.setText("")
        self.green.setText("")

class Engine(QObject):
    answer_ready = pyqtSignal(str, str)  # (context_black, answer_green)
    status_changed = pyqtSignal(str)
    _req_pause = pyqtSignal(str)
    _req_resume = pyqtSignal(str)
    _req_set_bbox = pyqtSignal(object)
    _req_set_interval = pyqtSignal(int)
    ocr_pulse = pyqtSignal()

    def __init__(self, bbox: CaptureRect,
                 min_interval_ms: int = None,
                 max_interval_ms: int = None):
        super().__init__()
        # pacing
        self._min_interval = int(os.getenv("MM_MIN_INTERVAL_MS", min_interval_ms or 1500))
        self._max_interval = int(os.getenv("MM_MAX_INTERVAL_MS", max_interval_ms or 5000))
        self._interval = self._min_interval

        self._bbox = bbox
        self._timer = QTimer(self)
        self._timer.setInterval(self._interval)
        self._timer.timeout.connect(self._tick)

        self._busy = False
        self._last_text = ""
        self._paused = False
        self.set_status = lambda s: self.status_changed.emit(s)

        # queued connections ensure these slots run in the Qt (main) thread
        self._req_pause.connect(self._do_pause, Qt.ConnectionType.QueuedConnection)
        self._req_resume.connect(self._do_resume, Qt.ConnectionType.QueuedConnection)
        self._req_set_bbox.connect(self._do_set_bbox, Qt.ConnectionType.QueuedConnection)
        self._req_set_interval.connect(self._do_set_interval, Qt.ConnectionType.QueuedConnection)
        self._gen = 0  # increments on every pause to invalidate in-flight work

    # ---- public controls ----
    # ---- capture loop ----
    # ---- capture loop ----
    def _tick(self):
        if self._busy or self._paused:
            return
        self._busy = True
        try:
            self.set_status(f"capturing… ({self._interval} ms)")
            screen = QGuiApplication.primaryScreen()
            pix = screen.grabWindow(0, self._bbox.left, self._bbox.top, self._bbox.width, self._bbox.height)
            qimg = pix.toImage()

            # 🔔 fire the pulse right as an OCR cycle begins
            self.ocr_pulse.emit()

            # capture current generation for cancellation
            gen = self._gen
        except Exception as e:
            self.answer_ready.emit("", f"(capture error: {e})")
            self.set_status("capture error")
            self._busy = False
            return

        # run OCR/LLM in background, with gen for cancelation
        threading.Thread(target=self._ocr_and_ask, args=(qimg, gen), daemon=True).start()

    def start(self):
        if not self._paused:
            self._timer.start()

    def stop(self):
        self._timer.stop()

    def pause(self, reason="paused"):
        self._req_pause.emit(reason)

    def resume(self, reason="resumed"):
        self._req_resume.emit(reason)

    def set_bbox(self, bbox: CaptureRect):
        self._req_set_bbox.emit(bbox)

        def capture_now(self):
            """Trigger an immediate capture/OCR cycle if idle."""
            if not self._busy and not self._paused:
                self._tick()

    def set_interval_ms(self, ms: int):
        self._req_set_interval.emit(ms)

    # ---- slots (main thread only) ----
    @pyqtSlot(str)
    def _do_pause(self, reason):
        self._paused = True
        self._gen += 1  # << invalidate any in-flight OCR/LLM
        self._timer.stop()
        self.set_status(f"{reason} ({self._interval} ms)")

    @pyqtSlot(str)
    def _do_resume(self, reason):
        self._paused = False
        self._timer.start()
        self.set_status(f"{reason} ({self._interval} ms)")

    @pyqtSlot(object)
    def _do_set_bbox(self, bbox: CaptureRect):
        self._bbox = bbox
        self._maybe_reset_to_min("⏸️ adjusted 📏")

    @pyqtSlot(int)
    def _do_set_interval(self, ms: int):
        self._interval = max(50, min(int(ms), self._max_interval))
        self._timer.setInterval(self._interval)
        self.set_status(f"⏸️⏲️ interval {self._interval} ms")

    # ---- pacing helpers (thread-safe wrappers) ----
    def _maybe_backoff(self, why: str):
        new_int = int(min(self._max_interval, max(self._interval, int(self._interval * 1.6))))
        if new_int != self._interval:
            self._req_set_interval.emit(new_int)
        self.set_status(f"{why} — backoff to {max(new_int, self._interval)} ms")

    def _maybe_reset_to_min(self, why: str):
        if self._interval != self._min_interval:
            self._req_set_interval.emit(self._min_interval)
            self.set_status(f"{why} — reset to {self._min_interval} ms")

    def _ocr_and_ask(self, qimg: QImage, gen: int):
        # cancel if a pause happened after this job started
        if gen != self._gen:
            self._busy = False
            return

        self.set_status("OCR…")

        # ---- OCR prep
        img = pil_from_qimage(qimg).convert("L")
        img = ImageOps.autocontrast(img, cutoff=2)
        img = img.filter(ImageFilter.MedianFilter(3))

        # cancel before OCR too
        if gen != self._gen:
            self._busy = False
            return

        # ---- OCR
        try:
            text = pytesseract.image_to_string(img, config="--oem 3 --psm 6 -l eng").strip()
        except Exception as e:
            if gen == self._gen:
                self.answer_ready.emit("", f"(OCR error: {e})")
                self.set_status("OCR error")
            self._busy = False
            return

        # cancel after OCR
        if gen != self._gen:
            self._busy = False
            return

        # pacing / empty text handling
        if not text:
            self.answer_ready.emit("", "")
            self.set_status("no text")
            self._last_text = ""
            self._maybe_backoff("no text")
            self._busy = False
            return

        if text == self._last_text:
            self._maybe_backoff("no change")
            self._busy = False
            return

        self._maybe_reset_to_min("new text")
        self.set_status(f"OCR… ({len(text)} chars)")
        self._last_text = text

        # ---- LLM call
        answer = ""  # <-- ensure defined even if something fails below
        if gen != self._gen:
            self._busy = False
            return

        try:
            self.set_status("asking…")
            prompt = (
                "Scan this text below for any questions, If there is no question, return one short helpful sentence."
                "If the question seems like it is a multiple-choice question answer as if it were a multiple-choice question."
                "You are a research assistant. For every question: "
                "1. Do not guess — if the answer is uncertain, state that clearly."
                "2. Research broadly across multiple reliable sources (e.g., academic, government, encyclopedic, technical references)."
                "3. If sources conflict, explain both sides and highlight the consensus view."
                "4. Provide a clear, structured final answer."
                "5. Include citations where possible."
                "6. Keep answers accurate, neutral, and free of unnecessary speculation."
                "7. If possible try to sum the answer up in 3 or fewer words. Under that always include the confidence rate of your answer in a percentage."
                "\n\n---\n"
                f"{text}\n---"
            )
            print("\n[MirrorMirror] Sending to ChatGPT model:")
            print("Model:", MODEL)
            print("Prompt:\n", prompt)
            sys.stdout.flush()

            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            answer = resp.choices[0].message.content.strip()
            self.set_status("✅ Answer Received!")
        except Exception as e:
            answer = f"(OpenAI error: {e})"
            self.set_status("LLM error")

        # cancel after LLM if we were paused mid-flight
        if gen != self._gen:
            self._busy = False
            return

        # final emit
        self.answer_ready.emit("", answer)
        self._busy = False


# ================== Header window ==================
class HeaderWindow(QWidget):
    OUTER_MARGINS = (16, 18, 16, 16)
    CARD_SIZE     = (440, 320)
    CARD_RADIUS   = 22

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mirror Mirror")
        self.setStyleSheet(
            """
               QWidget { background:#f4f4f6; }
               QFrame#pill {
                   background:#ffffff;
                   border-radius:16px;
                   border:1px solid #d0d0d0;
               }
               QLabel#instr {
                   color:#333333;
                   font-size:15px;
                   padding:8px 14px;
               }
               QLabel#status {
                   color:#555555;
                   font-size:13px;
                   padding:8px 10px;
               }
               QPushButton {
                   background-color:#0078d4;
                   color:white;
                   border:none;
                   border-radius:8px;
                   padding:6px 12px;
               }
               QPushButton:hover { background-color:#005fa3; }
               QPushButton:pressed { background-color:#004a82; }
           """
        )

        # --- header UI ---
        outer = QVBoxLayout(self);
        L, T, R, B = self.OUTER_MARGINS
        outer.setContentsMargins(L, T, R, B);
        outer.setSpacing(10)
        pill = QFrame()
        pill.setObjectName("pill")
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(12)
        shadow.setOffset(0, 2)
        pill.setGraphicsEffect(shadow)
        pill_l = QHBoxLayout(pill);
        pill_l.setContentsMargins(14, 6, 14, 6)
        self.instr = QLabel("Drag anywhere inside the gold pane to move · Place over text for an answer.")
        self.instr.setObjectName("instr")
        spacer = QSpacerItem(20, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        settings_btn = QPushButton("Settings")
        pill_l.addWidget(settings_btn)
        self.status_lbl = QLabel("idle");
        self.status_lbl.setObjectName("status")
        pill_l.addWidget(self.instr);
        pill_l.addItem(spacer)
        pill_l.addWidget(self.status_lbl, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        settings_btn.clicked.connect(self._open_settings)
        outer.addWidget(pill, 0)

        # green ANSWER in header
        self.answer_lbl = QLabel("")
        self.answer_lbl.setStyleSheet("color:#1aa34a; font-size:16px; font-weight:600; padding:6px 10px;")
        self.answer_lbl.setWordWrap(True)
        self.answer_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self.answer_lbl, 0)

        # header window size/pos
        self.adjustSize()
        w_card, h_card = self.CARD_SIZE
        self.resize(w_card + L + R, self.height())
        self.move(140, 120)

        # pause debounce timer
        self._pause_timer = QTimer(self)
        self._pause_timer.setSingleShot(True)
        self._pause_timer.timeout.connect(self._resume_engine_after_pause)

        self._glass = None
        self._engine = None
        self._settings = None

        # ---------------------------
        # 1) Create GLASS first
        # ---------------------------
        self._place_glass()

        # ---------------------------
        # 2) Create ENGINE next
        # ---------------------------
        screen = QGuiApplication.primaryScreen()
        geo = screen.geometry()
        cap_left = (geo.width() - 800) // 2
        cap_top = (geo.height() - 400) // 2
        bbox = CaptureRect(cap_left, cap_top, 800, 400)

        self._engine = Engine(bbox)

        # ---------------------------
        # 3) Wire ENGINE → UI/Glass
        # ---------------------------
        self._engine.status_changed.connect(self.set_status, type=Qt.ConnectionType.QueuedConnection)
        self._engine.answer_ready.connect(self._on_answer, type=Qt.ConnectionType.QueuedConnection)
        self._engine.ocr_pulse.connect(self._glass.trigger_glow, type=Qt.ConnectionType.QueuedConnection)

        # ---------------------------
        # 4) Wire GLASS → Engine/UI
        # ---------------------------
        self._glass.pauseRequested.connect(self._on_pause_request, type=Qt.ConnectionType.QueuedConnection)
        self._glass.adjusting.connect(self._on_glass_adjusting, type=Qt.ConnectionType.QueuedConnection)
        self._glass.geometry_changed.connect(self._on_glass_geometry_changed, type=Qt.ConnectionType.QueuedConnection)
        self._glass.ocrRequested.connect(self._engine.capture_now, type=Qt.ConnectionType.QueuedConnection)

        # sync initial geometry and start
        self._glass.setGeometry(QRect(bbox.left, bbox.top, bbox.width, bbox.height))
        self.set_status(f"ready (tess: {'✅' if os.path.exists(_TESS) else 'missing'})")
        self._engine.start()

    def _on_pause_request(self, should_pause: bool):
        if not self._engine:
            return
        if should_pause:
            self._engine.pause("⏸️ paused (dragging ✊)")
            self._pause_timer.stop()
        else:
            # resume shortly after final release (debounced)
            self._pause_timer.start(350)

    def set_status(self, msg: str):
        if self.status_lbl:
            self.status_lbl.setText(msg or "")

    def _on_answer(self, black: str, green: str):
        # Glass remains textless; header shows the answer in green
        self.answer_lbl.setText(green or "")

    def _place_glass(self):
        L,T,R,B = self.OUTER_MARGINS
        w_card, h_card = self.CARD_SIZE
        top_left = self.mapToGlobal(QPoint(L, self.height() + 8))
        g = QRect(top_left, QPoint(top_left.x() + w_card, top_left.y() + h_card))
        self._glass = GlassPane(g, corner_radius=self.CARD_RADIUS)
        self._glass.show()

    # pause/resume around header moves/resizes
    def moveEvent(self, e):
        super().moveEvent(e)
        if self._engine:
            self._engine.pause("⏸️ paused (moving header 📏)")
            self._pause_timer.start(350)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._engine:
            self._engine.pause("⏸️ paused (resizing header 📏)")
            self._pause_timer.start(350)

    def _resume_engine_after_pause(self):
        if self._engine:
            self._engine.resume("▶️ resumed")

    def _open_settings(self):
        """Close header and show settings window."""
        if self._engine:
            self._engine.stop()
        if self._glass:
            self._glass.close()

        def reopen():
            w = HeaderWindow()
            return w

        self._settings = SettingsWindow(reopen)
        self._settings.show()
        self.close()

    # glass hooks
    def _on_glass_adjusting(self, is_adjusting: bool):
        if not self._engine: return
        if is_adjusting:
            self._engine.pause("⏸️ paused (adjusting 📏)")
            self._pause_timer.stop()
        else:
            self._pause_timer.start(350)

    def _on_glass_geometry_changed(self, new_geo: QRect):
        if not self._engine:
            return

            # --- NEW: pause OCR on any geometry change (move/resize from ANY side) ---
        self._engine.pause("⏸️ paused (adjusting 📏)")
        self._pause_timer.stop()
        self._pause_timer.start(350)  # resume a bit after the last change
        # -------------------------------------------------------------------------

        bbox = CaptureRect(new_geo.x(), new_geo.y(), new_geo.width(), new_geo.height())
        self._engine.set_bbox(bbox)
        self.set_status("adjusted 📏")

# ================== Welcome window ==================
class WelcomeWindow(QWidget):
    """Initial window presenting a friendly welcome and basic instructions."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mirror Mirror")
        layout = QVBoxLayout(self)

        msg = (
            "Welcome to Mirror Mirror!\n\n"
            "This app watches a portion of your screen, uses OCR and OpenAI to read it, "
            "and shows a helpful answer in the green header."
        )
        lbl = QLabel(msg)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        btn_row = QHBoxLayout()
        layout.addLayout(btn_row)

        start_btn = QPushButton("Start")
        settings_btn = QPushButton("Settings")
        quit_btn = QPushButton("Quit")
        btn_row.addStretch(1)
        btn_row.addWidget(start_btn)
        btn_row.addWidget(settings_btn)
        btn_row.addWidget(quit_btn)

        start_btn.clicked.connect(self._launch)
        settings_btn.clicked.connect(self._open_settings)
        quit_btn.clicked.connect(QApplication.instance().quit)

        self._main = None
        self._settings = None

    def _launch(self):
        """Close welcome window and show the main HeaderWindow."""
        self._main = HeaderWindow()
        self._main.show()
        self.close()

    def _open_settings(self):
        """Close welcome window and show settings."""
        def reopen():
            return WelcomeWindow()

        self._settings = SettingsWindow(reopen)
        self._settings.show()
        self.close()

# ================== Settings window ==================
class SettingsWindow(QWidget):
    """Simple window for adjusting application settings."""

    def __init__(self, return_factory=None):
        super().__init__()
        self.setWindowTitle("Settings")
        layout = QVBoxLayout(self)
        #===VSettings bannerV===
        lbl = QLabel("")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        color_row = QHBoxLayout()
        layout.addLayout(color_row)

        self._color_indicator = QLabel()
        self._color_indicator.setFixedSize(24, 24)
        color_row.addWidget(self._color_indicator)

        self._color_btn = QPushButton("Choose Pane Color")
        color_row.addWidget(self._color_btn)

        self._reset_btn = QPushButton("Reset to Default")
        color_row.addWidget(self._reset_btn)

        self._update_color_display()
        self._color_btn.clicked.connect(self._choose_color)
        self._reset_btn.clicked.connect(self._reset_color)

        back_btn = QPushButton("Back")
        layout.addWidget(back_btn, 0, Qt.AlignmentFlag.AlignRight)

        self._return_factory = return_factory
        self._return_widget = None
        back_btn.clicked.connect(self._go_back)

    def _update_color_display(self):
        self._color_indicator.setStyleSheet(
            f"border-radius: 12px; background-color: {PANE_COLOR.name()};"
        )

    def _choose_color(self):
        global PANE_COLOR
        color = QColorDialog.getColor(PANE_COLOR, self, "Select Pane Color")
        if color.isValid():
            PANE_COLOR = QColor(color.red(), color.green(), color.blue(), 230)
            self._update_color_display()

    def _reset_color(self):
        global PANE_COLOR
        PANE_COLOR = QColor(DEFAULT_PANE_COLOR)
        self._update_color_display()

    def _go_back(self):
        if self._return_factory:
            self._return_widget = self._return_factory()
            self._return_widget.show()
        self.close()


# ================== Welcome window ==================
class WelcomeWindow(QWidget):
    """Initial window presenting a friendly welcome and basic instructions."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mirror Mirror")
        layout = QVBoxLayout(self)

        msg = (
            "Welcome to Mirror Mirror!\n\n"
            "This app watches a portion of your screen, uses OCR and OpenAI to read it, "
            "and shows a helpful answer in the green header."
        )
        lbl = QLabel(msg)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        btn_row = QHBoxLayout()
        layout.addLayout(btn_row)

        start_btn = QPushButton("Start")
        settings_btn = QPushButton("Settings")
        quit_btn = QPushButton("Quit")
        btn_row.addStretch(1)
        btn_row.addWidget(start_btn)
        btn_row.addWidget(settings_btn)
        btn_row.addWidget(quit_btn)

        start_btn.clicked.connect(self._launch)
        settings_btn.clicked.connect(self._open_settings)
        quit_btn.clicked.connect(QApplication.instance().quit)

        self._main = None
        self._settings = None

    def _launch(self):
        """Close welcome window and show the main HeaderWindow."""
        self._main = HeaderWindow()
        self._main.show()
        self.close()

    def _open_settings(self):
        """Close welcome window and show settings."""
        def reopen():
            return WelcomeWindow()

        self._settings = SettingsWindow(reopen)
        self._settings.show()
        self.close()

# ================== Welcome window ==================
class WelcomeWindow(QWidget):
    """Initial window presenting a friendly welcome and basic instructions."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mirror Mirror")
        layout = QVBoxLayout(self)

        msg = (
            "Welcome to Mirror Mirror!\n\n"
            "This app watches a portion of your screen, uses OCR and OpenAI to read it, "
            "and shows a helpful answer in the green header."
        )
        lbl = QLabel(msg)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        btn_row = QHBoxLayout()
        layout.addLayout(btn_row)

        start_btn = QPushButton("Start")
        settings_btn = QPushButton("Settings")
        quit_btn = QPushButton("Quit")
        btn_row.addStretch(1)
        btn_row.addWidget(start_btn)
        btn_row.addWidget(settings_btn)
        btn_row.addWidget(quit_btn)

        start_btn.clicked.connect(self._launch)
        settings_btn.clicked.connect(self._open_settings)
        quit_btn.clicked.connect(QApplication.instance().quit)

        self._main = None
        self._settings = None

    def _launch(self):
        """Close welcome window and show the main HeaderWindow."""
        self._main = HeaderWindow()
        self._main.show()
        self.close()

    def _open_settings(self):
        """Close welcome window and show settings."""
        def reopen():
            return WelcomeWindow()

        self._settings = SettingsWindow(reopen)
        self._settings.show()
        self.close()

# ================== main ==================
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        app = QApplication(sys.argv)
        QMessageBox.critical(None, "Mirror Mirror", "Set OPENAI_API_KEY (e.g., in a .env file).")
        sys.exit(1)

    app = QApplication(sys.argv)
    w = WelcomeWindow()
    w.show()
    sys.exit(app.exec())
