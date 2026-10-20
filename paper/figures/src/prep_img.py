"""Crop the lavapipe captures to 4:3 tiles for the figures (JPEG, 2x the printed size)."""
import sys
from pathlib import Path
from PIL import Image
W = Path(__file__).resolve().parent.parent
crops = {  # name -> (source, crop box or None)
}
def tile(src, dst, box=None, w=480):
    im = Image.open(src).convert('RGB')
    if box: im = im.crop(box)
    x, y = im.size
    if abs(x / y - 4 / 3) > 0.01:          # centre-crop to 4:3
        if x / y > 4 / 3: nx = round(y * 4 / 3); im = im.crop(((x - nx) // 2, 0, (x - nx) // 2 + nx, y))
        else: ny = round(x * 3 / 4); im = im.crop((0, (y - ny) // 2, x, (y - ny) // 2 + ny))
    im = im.resize((w, w * 3 // 4), Image.LANCZOS)
    im.save(dst, quality=92)
if __name__ == '__main__':
    for a in sys.argv[1:]:
        name, src, *box = a.split(':')
        tile(src, W / 'src' / 'img' / (name + '.jpg'), tuple(map(int, box[0].split(','))) if box else None)
