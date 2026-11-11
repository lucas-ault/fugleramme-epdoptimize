// epdoptimize 1.4.0, Apache-2.0 (LICENSE). RgbQuant is bundled in dist and is MIT
// (LICENSE.rgbquant). Reads a width, a height and RGBA on stdin. Writes device-color
// RGB on stdout. A choice of "auto" reports the classification on stderr as one
// line beginning "EPD ". A choice of "custom" takes a JSON options object as its
// third argument and spreads it over the named preset.

import { readFileSync } from "node:fs";
import {
  aitjcizeSpectra6Palette,
  ditherImage,
  replaceColors,
  suggestCanvasProcessingOptions,
} from "./dist/index.mjs";

function canvas(width, height, data) {
  const surface = {
    width,
    height,
    data,
    getContext() {
      return {
        getImageData() {
          return {
            width: surface.width,
            height: surface.height,
            data: new Uint8ClampedArray(surface.data),
          };
        },
        putImageData(image) {
          surface.width = image.width;
          surface.height = image.height;
          surface.data = new Uint8ClampedArray(image.data);
        },
      };
    },
  };
  return surface;
}

const preset = process.argv[2];
let extra = {};
if (process.argv[3]) {
  try {
    extra = JSON.parse(process.argv[3]);
  } catch {
    process.stderr.write("epdoptimize: bad options\n");
    process.exit(1);
  }
}
const input = readFileSync(0);
if (input.length < 8) {
  process.stderr.write("epdoptimize: short header\n");
  process.exit(1);
}
const width = input.readUInt32LE(0);
const height = input.readUInt32LE(4);
const pixels = width * height;
if (input.length !== 8 + pixels * 4) {
  process.stderr.write(`epdoptimize: expected ${pixels * 4} bytes, got ${input.length - 8}\n`);
  process.exit(1);
}

const source = canvas(
  width,
  height,
  new Uint8ClampedArray(input.buffer, input.byteOffset + 8, pixels * 4),
);
const dithered = canvas(width, height, new Uint8ClampedArray(pixels * 4));
const device = canvas(width, height, new Uint8ClampedArray(pixels * 4));
const palette = aitjcizeSpectra6Palette;

let options = { palette, processingEngine: "auto", adjustmentEngine: "js" };
if (preset === "auto") {
  const suggestion = suggestCanvasProcessingOptions(source, palette);
  const ranked = Object.entries(suggestion.scores ?? {}).sort((a, b) => b[1] - a[1]);
  const chosen = suggestion.ditherOptions.processingPreset ?? ranked[0]?.[0] ?? "";
  process.stderr.write(`EPD ${JSON.stringify({ imageKind: suggestion.imageKind, preset: chosen })}\n`);
  options = { ...suggestion.ditherOptions, palette, processingEngine: "auto", adjustmentEngine: "js" };
} else if (preset === "custom") {
  options = {
    ...options,
    ...extra,
    palette,
    processingEngine: "auto",
    adjustmentEngine: "js",
    processingPreset: extra.processingPreset || "balanced",
  };
} else {
  options.processingPreset = preset;
}

await ditherImage(source, dithered, options);
replaceColors(dithered, device, palette);

const rgba = device.data;
const rgb = Buffer.alloc(pixels * 3);
for (let i = 0; i < pixels; i += 1) {
  rgb[i * 3] = rgba[i * 4];
  rgb[i * 3 + 1] = rgba[i * 4 + 1];
  rgb[i * 3 + 2] = rgba[i * 4 + 2];
}
process.stdout.write(rgb);
