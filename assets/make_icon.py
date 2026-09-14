"""Generate assets/icon.icns: a gold glass frame on a frosted rounded square."""
import subprocess, shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).parent
SIZE = 1024

def render(size: int) -> Image.Image:
    s = size
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = int(s * 0.22)
    # frosted body with a soft vertical gradient
    body = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    bd = ImageDraw.Draw(body)
    for y in range(s):
        t = y / s
        bd.line([(0, y), (s, y)], fill=(int(28 + 30 * t), int(30 + 30 * t), int(40 + 36 * t), 255))
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle([int(s*0.06), int(s*0.06), int(s*0.94), int(s*0.94)], r, fill=255)
    img.paste(body, (0, 0), mask)
    # sheen
    sheen = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sheen)
    sd.ellipse([int(-s*0.2), int(-s*0.55), int(s*1.2), int(s*0.5)], fill=(255, 255, 255, 46))
    sheen = sheen.filter(ImageFilter.GaussianBlur(s * 0.03))
    img.alpha_composite(Image.composite(sheen, Image.new("RGBA", (s, s), (0, 0, 0, 0)), mask))
    # gold frame (the capture pane)
    d = ImageDraw.Draw(img)
    pad = int(s * 0.2); w = max(2, int(s * 0.045))
    d.rounded_rectangle([pad, int(s*0.26), s - pad, s - int(s*0.26)], int(s*0.1), outline=(212, 175, 55, 255), width=w)
    d.rounded_rectangle([pad + w, int(s*0.26) + w, s - pad - w, s - int(s*0.26) - w], int(s*0.08), outline=(255, 246, 200, 110), width=max(1, w // 4))
    # green answer dot
    cr = int(s * 0.05)
    cx, cy = s - pad - int(s*0.11), s - int(s*0.26) - int(s*0.11)
    d.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=(48, 209, 88, 255))
    return img

def main():
    iconset = HERE / "icon.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir()
    base = render(SIZE)
    for px in (16, 32, 128, 256, 512):
        base.resize((px, px), Image.LANCZOS).save(iconset / f"icon_{px}x{px}.png")
        base.resize((px * 2, px * 2), Image.LANCZOS).save(iconset / f"icon_{px}x{px}@2x.png")
    base.save(HERE / "icon_1024.png")
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(HERE / "icon.icns")], check=True)
    shutil.rmtree(iconset)
    print("wrote", HERE / "icon.icns")

if __name__ == "__main__":
    main()
