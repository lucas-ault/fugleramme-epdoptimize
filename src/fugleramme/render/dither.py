"""Panel color reduction.

The Inky Impression (Spectra 6) shows only 6 colors. The panel, the kiosk and
the admin preview are reduced here rather than in the driver, because
`inky.set_image` re-dithers anything that is not already a 6-color "P" image -
dithering an already-dithered image smears it. Handing it a "P" image with
exactly 6 palette entries makes it skip its own dither and just remap indices,
so this module owns the reduction and the saved frame matches the glass.

Index order below is the driver's (`inky.inky_el133uf1`), which the remap
depends on. The palette is a blend of the driver's two: the "desaturated" set is
what it writes, the "saturated" set is nearer what the panel actually shows, so
dithering against the blend measures color distance in roughly panel space.
"""

from __future__ import annotations

import json
import logging
import struct
import subprocess
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from ..config import REPO_ROOT
from .paper import PANEL_PAPER, _reach, paper_texture

log = logging.getLogger(__name__)

# What the admin offers. `frame` is this module's own reduction, and the value a
# settings file from before the choice names. The rest are epdoptimize's presets;
# `auto` asks that library to pick one from the page. `custom` starts from a
# preset and replaces only the fields a recipe names.
FRAME = "frame"
CUSTOM = "custom"
DITHERS: dict[str, tuple[str, str]] = {
    FRAME: (
        "Frame",
        "The panel's own reduction: the birds darkened, then Floyd-Steinberg onto the six inks.",
    ),
    "auto": ("Auto", "The page is classified and a preset is chosen for it."),
    "balanced": (
        "Balanced",
        "Compresses the display's brightness range. A general starting point.",
    ),
    "dynamic": ("Dynamic", "An S-curve, for a brighter, punchier picture."),
    "vivid": ("Vivid", "Boosts colour, with a gentler S-curve, for illustrations."),
    "soft": ("Soft", "Less contrast, and a smoother dither."),
    "grayscale": ("Grayscale", "Removes colour and matches in perceptual brightness."),
    "restore": ("Restore", "Opens up a faded scan or painting before mapping it to the inks."),
    "posterScan": (
        "Poster scan",
        "Cools warm paper, holds the black ink, and keeps strong poster colours.",
    ),
    CUSTOM: (
        "Custom",
        "Starts from a preset. The rows below replace only what you change.",
    ),
}

# Empty is "as the preset": that part is not sent, so the base preset keeps it.
PRESETS = {key: label for key, (label, _) in DITHERS.items() if key not in (FRAME, "auto", CUSTOM)}
DITHER_TYPES = {
    "": "As the preset",
    "errorDiffusion": "Error diffusion",
    "ordered": "Ordered",
    "blueNoise": "Blue noise",
    "quantizationOnly": "Plain",
}
DITHER_MATRICES = {
    "": "As the preset",
    "floydSteinberg": "Floyd-Steinberg",
    "atkinson": "Atkinson",
    "jarvis": "Jarvis",
    "stucki": "Stucki",
    "burkes": "Burkes",
    "sierra3": "Sierra",
}
DITHER_MATCHES = {"": "As the preset", "rgb": "RGB", "lab": "LAB", "chroma": "Chroma"}
DITHER_TONES = {"": "As the preset", "off": "Off", "contrast": "Contrast", "scurve": "S-curve"}
DITHER_RANGES = {"": "As the preset", "off": "Off", "display": "Display", "auto": "Auto"}
_DIFFUSION = ("", "errorDiffusion")

# Bookworm's nodejs package is 18, which cannot run the bundle. NodeSource's 20.x can.
NODE_INSTALL = "curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt-get install -y nodejs"
_SCRIPT = REPO_ROOT / "vendor" / "epdoptimize" / "dither.mjs"
_node_ok = False
_last_failure: str | None = None
_last_auto: tuple[str, str] | None = None

# Driver index order: black, white, yellow, red, blue, green.
_DESATURATED = [(0, 0, 0), (255, 255, 255), (255, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0)]
_SATURATED = [(0, 0, 0), (161, 164, 165), (208, 190, 71), (156, 72, 75), (61, 59, 94), (58, 91, 70)]

BLACK = 0  # the index, for painting straight onto a dithered page

# Matches the inky default. Raising it dithers against duller targets, which
# spreads more ink; the blend must stay nearest-neighbour to _DESATURATED or the
# driver's remap will land on the wrong color.
SATURATION = 0.5

_LIFT = np.array([0.887, 0.906, 0.950])
_PIVOT = 182
_CONTRAST = 1.15
_FADE = 4
_HALO = 12
_HALO_REACH = 32

PALETTE_6 = [
    tuple(int(s * SATURATION + d * (1.0 - SATURATION)) for s, d in zip(sat, desat, strict=True))
    for sat, desat in zip(_SATURATED, _DESATURATED, strict=True)
]


def _palette_image() -> Image.Image:
    pal = Image.new("P", (1, 1))
    flat: list[int] = []
    for colour in PALETTE_6:
        flat += list(colour)
    # Pad with black, already index 0, so the palette still reads as 6 unique colors.
    flat += [0, 0, 0] * (256 - len(PALETTE_6))
    pal.putpalette(flat)
    return pal


def _boost(rgb: np.ndarray) -> np.ndarray:
    return _PIVOT + (rgb * _LIFT - _PIVOT) * _CONTRAST


def _pop(image: Image.Image) -> Image.Image:
    """Darken the birds so pale plumage takes ink. The paper and the halos keep
    their tone, or the halos would ring the birds."""
    rgb = np.asarray(image.convert("RGB"), np.float64)
    paper = np.array(PANEL_PAPER, np.float64)
    dist = np.linalg.norm(rgb - paper, axis=2)
    halo = _reach(dist <= 1, dist <= _HALO, _HALO_REACH) > 0
    keep = np.maximum(np.exp(-dist / _FADE), halo)[..., None]
    out = _boost(rgb) + keep * (paper - _boost(paper))
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


def _frame(image: Image.Image) -> Image.Image:
    return _pop(image).quantize(palette=_palette_image(), dither=Image.Dither.FLOYDSTEINBERG)


def _paper_mask(image: Image.Image) -> np.ndarray:
    """The bare sheet and the halo around each bird. The web page is the grained
    sheet and the panel page is flat, and either one is most of the pixels, so a
    preset's curve lifts the page to white unless these are put back."""
    rgb = np.asarray(image.convert("RGB"), np.float64)
    flat = np.linalg.norm(rgb - np.array(PANEL_PAPER), axis=2)
    grained = np.asarray(paper_texture(*image.size), np.float64)
    dist = np.minimum(flat, np.linalg.norm(rgb - grained, axis=2))
    bare = dist <= 1
    return bare | (_reach(bare, dist <= _HALO, _HALO_REACH) > 0)


def _restore_paper(source: Image.Image, reduced: Image.Image) -> Image.Image:
    """The preset's inks for the birds, the frame reduction for the paper. Both
    share the driver's index order. The frame's palette is what a preview shows
    for an ink, and it still remaps onto the driver as the identity."""
    framed = _frame(source)
    mask = _paper_mask(source)
    out = np.array(reduced)
    out[mask] = np.asarray(framed)[mask]
    page = Image.fromarray(out, "P")
    page.putpalette(framed.getpalette() or [])
    return page


def exposure_label(value: float) -> str:
    """How an exposure slider reads. `0` is the picture as it is."""
    value = round(value, 1)
    if value == 0:
        return "Neutral"
    stops = abs(value)
    unit = "stop" if stops == 1 else "stops"
    way = "brighter" if value > 0 else "darker"
    return f"{stops:g} {unit} {way}"


def saturation_label(value: float) -> str:
    value = round(value, 1)
    if value == 0:
        return "Neutral"
    much = abs(value) >= 0.5
    if value > 0:
        return "Much richer" if much else "Richer"
    return "Much paler" if much else "Paler"


def contrast_label(value: float) -> str:
    value = round(value, 1)
    if value == 0:
        return "Neutral"
    much = abs(value) >= 0.5
    if value > 0:
        return "Much more" if much else "More"
    return "Much less" if much else "Less"


def curve_label(value: float) -> str:
    value = round(value, 1)
    if value <= 0:
        return "Flat"
    if value < 0.4:
        return "Gentle"
    if value < 0.8:
        return "Medium"
    return "Strong"


@dataclass(frozen=True)
class Recipe:
    """A Custom reduction. An empty choice stays with `base`, so a file that
    names none of these is Balanced with nothing replaced."""

    base: str = "balanced"
    kind: str = ""
    matrix: str = ""
    serpentine: bool = False
    match: str = ""
    tone: str = ""
    exposure: float = 0
    saturation: float = 0
    contrast: float = 0
    curve: float = 0.7
    range: str = ""

    def options(self) -> dict[str, Any]:
        """What epdoptimize is given. Only a choice that differs from the preset."""
        out: dict[str, Any] = {"processingPreset": self.base}
        if self.kind:
            out["ditheringType"] = self.kind
        if self.kind in _DIFFUSION:
            if self.matrix:
                out["errorDiffusionMatrix"] = self.matrix
            if self.serpentine:
                out["serpentine"] = True
        if self.match:
            out["colorMatching"] = self.match
        tone = _tone(self)
        if tone is not None:
            out["toneMapping"] = tone
        if self.range:
            out["dynamicRangeCompression"] = {"mode": self.range}
        return out


def _tone(recipe: Recipe) -> dict[str, Any] | None:
    if not recipe.tone:
        return None
    if recipe.tone == "off":
        return {"mode": "off"}
    if recipe.tone == "contrast":
        return {
            "mode": "contrast",
            "exposure": recipe.exposure,
            "saturation": recipe.saturation,
            "contrast": recipe.contrast,
        }
    return {
        "mode": "scurve",
        "exposure": recipe.exposure,
        "saturation": recipe.saturation,
        "strength": recipe.curve,
    }


def _device_palette() -> Image.Image:
    """The driver's six inks, so a no-dither quantize lands on its index order."""
    pal = Image.new("P", (1, 1))
    flat: list[int] = []
    for colour in _DESATURATED:
        flat += list(colour)
    flat += [0, 0, 0] * (256 - len(_DESATURATED))
    pal.putpalette(flat)
    return pal


def node_ready() -> bool:
    """Whether `node` is present and new enough to run the vendored bundle.
    A miss is checked again next time; a hit sticks for the process."""
    global _node_ok
    if _node_ok:
        return True
    try:
        done = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    major = done.stdout.strip().lstrip("v").split(".", 1)[0]
    _node_ok = done.returncode == 0 and major.isdigit() and int(major) >= 20
    return _node_ok


def _failed(reason: str, image: Image.Image) -> Image.Image:
    """The frame dither, logged once per distinct cause so a missing Node does not
    fill the journal on every render."""
    global _last_failure
    if reason != _last_failure:
        log.warning("Panel colour preset unavailable (%s); using the frame dither", reason)
        _last_failure = reason
    return _frame(image)


def _note_auto(stderr: bytes) -> None:
    """Auto names its classification when that choice changes, the way a missing
    plate is logged once."""
    global _last_auto
    line = next(
        (item for item in stderr.decode(errors="replace").splitlines() if item.startswith("EPD ")),
        "",
    )
    try:
        choice = json.loads(line.removeprefix("EPD "))
    except json.JSONDecodeError:
        return
    key = (str(choice.get("imageKind", "")), str(choice.get("preset", "")))
    if key == _last_auto:
        return
    _last_auto = key
    log.info("Auto dither chose %s for a %s page", key[1], key[0])


def _epd(image: Image.Image, preset: str, options: dict[str, Any] | None = None) -> Image.Image:
    """epdoptimize against its Spectra 6 palette, then the driver's inks with no
    second dither. The paper and the halo are put back from the frame reduction,
    so the preset's curve does not lift the sheet to white. A failure keeps the
    page moving on the frame dither. `options` is a Custom recipe; a preset
    passes none."""
    global _last_failure
    rgba = image.convert("RGBA")
    width, height = rgba.size
    command = ["node", str(_SCRIPT), preset]
    if options:
        command.append(json.dumps(options, separators=(",", ":")))
    try:
        done = subprocess.run(
            command,
            input=struct.pack("<II", width, height) + rgba.tobytes(),
            capture_output=True,
            check=True,
            timeout=120,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace").strip().splitlines()
        return _failed(detail[-1] if detail else f"exit {exc.returncode}", image)
    except (OSError, subprocess.SubprocessError) as exc:
        return _failed(str(exc), image)
    if len(done.stdout) != width * height * 3:
        return _failed(f"short reply, {len(done.stdout)} bytes", image)
    _last_failure = None
    if preset == "auto":
        _note_auto(done.stderr)
    rgb = Image.frombytes("RGB", (width, height), done.stdout)
    reduced = rgb.quantize(palette=_device_palette(), dither=Image.Dither.NONE)
    return _restore_paper(image, reduced)


def dither(image: Image.Image, preset: str = FRAME, recipe: Recipe | None = None) -> Image.Image:
    """Quantize an RGB image to the 6-color panel palette (returns mode "P").
    `recipe` applies only when `preset` is Custom."""
    if preset not in DITHERS or preset == FRAME:
        return _frame(image)
    if preset == CUSTOM:
        return _epd(image, preset, (recipe or Recipe()).options())
    return _epd(image, preset)


def monochrome(image: Image.Image) -> Image.Image:
    """A black and white page onto the panel palette as it is, undithered: the setup screen's
    QR code stops scanning once it is smeared."""
    return image.convert("RGB").quantize(palette=_palette_image(), dither=Image.Dither.NONE)
