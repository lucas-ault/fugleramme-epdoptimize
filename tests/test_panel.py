"""Panel push invariants. The driver is Pi-only, so a fake device stands in for
the two things this side has to get right off-hardware: the buffer handed to
`set_image` is always the panel's native landscape, and the dithered image is
one the driver will pass through instead of dithering a second time."""

from __future__ import annotations

import shutil
import subprocess

import numpy
import pytest
from PIL import Image

from fugleramme.panel import Panel, external_resolution, resolution_of
from fugleramme.render.dither import dither
from fugleramme.render.paper import PANEL_PAPER, paper_texture

# inky.inky_el133uf1.DESATURATED_PALETTE - what the driver quantizes against.
DESATURATED = [
    [0, 0, 0],
    [255, 255, 255],
    [255, 255, 0],
    [255, 0, 0],
    [0, 0, 255],
    [0, 255, 0],
    [255, 255, 255],
]


class FakeInky:
    resolution = (1600, 1200)

    def __init__(self):
        self.image = None
        self.shown = 0

    def set_image(self, image):
        if image.size != self.resolution:
            raise ValueError("wrong size")
        self.image = image

    def show(self):
        self.shown += 1


def _panel() -> tuple[Panel, FakeInky]:
    device = FakeInky()
    return Panel(device), device


def test_unrotated_image_pushes_as_is():
    panel, device = _panel()
    panel.push(Image.new("P", (1600, 1200)))
    assert device.image.size == (1600, 1200)
    assert device.shown == 1


@pytest.mark.parametrize(
    "rotation,size", [(90, (1200, 1600)), (180, (1600, 1200)), (270, (1200, 1600))]
)
def test_every_rotation_lands_in_the_native_buffer(rotation, size):
    panel, device = _panel()
    panel.push(Image.new("RGB", size), rotation)
    assert device.image.size == panel.resolution


def test_rotation_direction_is_counter_clockwise():
    # 90 is the case verified on hardware; the others are derived from it.
    panel, device = _panel()
    source = Image.new("RGB", (1200, 1600))
    source.putpixel((0, 0), (255, 0, 0))  # top-left -> bottom-left when rotated CCW
    panel.push(source, 90)
    assert device.image.getpixel((0, 1199)) == (255, 0, 0)


def test_mismatched_size_raises_rather_than_pushing():
    panel, device = _panel()
    with pytest.raises(ValueError):
        panel.push(Image.new("P", (800, 480)))
    assert device.shown == 0


def test_rotation_that_does_not_fit_the_panel_raises():
    # A portrait render sent without its rotation would otherwise be silently wrong.
    panel, device = _panel()
    with pytest.raises(ValueError):
        panel.push(Image.new("P", (1200, 1600)))
    assert device.shown == 0


def test_dither_survives_the_driver_untouched():
    # The driver re-dithers unless the image is "P" with exactly 6 palette
    # colours, and then remaps by nearest DESATURATED colour - which must be the
    # identity, or the panel shows the wrong ink.
    noise = Image.effect_noise((160, 120), 90).convert("L")
    source = Image.merge("RGB", (noise, noise.rotate(37), noise.rotate(90)))
    image = dither(source)
    assert image.mode == "P"
    assert len(image.palette.colors) == 6

    palette_image = Image.new("P", (1, 1))
    palette_image.putpalette(numpy.array(DESATURATED, dtype=numpy.uint8).flatten().tobytes())
    remapped = image.convert("RGB").quantize(6, palette=palette_image, dither=Image.Dither.NONE)

    assert numpy.array_equal(numpy.array(image), numpy.array(remapped))


def _node_20() -> None:
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    try:
        version = subprocess.check_output(["node", "-p", "process.versions.node"], text=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("node did not answer")
    major = version.strip().split(".", 1)[0]
    if not major.isdigit() or int(major) < 20:
        pytest.skip("node 20 is required")


def test_a_preset_survives_the_driver_the_same_way():
    # The library dithers against its own palette and this side remaps onto the
    # driver's, with no second dither. Without Node 20 the test cannot run.
    _node_20()

    noise = Image.effect_noise((80, 60), 90).convert("L")
    source = Image.merge("RGB", (noise, noise.rotate(37), noise.rotate(90)))
    image = dither(source, "balanced")
    assert image.mode == "P"
    assert len(image.palette.colors) == 6
    palette_image = Image.new("P", (1, 1))
    palette_image.putpalette(numpy.array(DESATURATED, dtype=numpy.uint8).flatten().tobytes())
    remapped = image.convert("RGB").quantize(6, palette=palette_image, dither=Image.Dither.NONE)
    assert numpy.array_equal(numpy.array(image), numpy.array(remapped))


def test_a_preset_keeps_the_frames_paper():
    # Grayscale would pull the sheet with the birds. The sheet and its halo stay
    # on the frame reduction; the bird does not.
    _node_20()
    source = Image.new("RGB", (48, 36), PANEL_PAPER)
    source.paste(Image.new("RGB", (12, 10), (20, 40, 180)), (18, 13))
    page = dither(source, "grayscale")
    framed = dither(source)
    assert page.convert("RGB").getpixel((0, 0)) == framed.convert("RGB").getpixel((0, 0))
    assert page.getpixel((24, 18)) != framed.getpixel((24, 18))

    textured = paper_texture(48, 36)
    textured.paste(Image.new("RGB", (12, 10), (20, 40, 180)), (18, 13))
    page = dither(textured, "grayscale")
    framed = dither(textured)
    assert page.convert("RGB").getpixel((0, 0)) == framed.convert("RGB").getpixel((0, 0))
    assert page.getpixel((24, 18)) != framed.getpixel((24, 18))


def test_the_external_panel_takes_its_pick_else_the_inky_s_size_else_the_13_inch():
    device = FakeInky()
    device.resolution = (800, 480)
    inky = Panel(device)
    assert external_resolution(inky, "") == (800, 480)
    assert external_resolution(inky, "4") == (640, 400)
    assert external_resolution(None, "") == (1600, 1200)
    assert resolution_of(inky, "4") == (800, 480)  # this frame's own page stays the Inky's
