"""
Dependency-light PDF report builder built on Pillow.

We deliberately avoid matplotlib/reportlab here: this pipeline targets
**Windows on ARM64 (Snapdragon)** where Python 3.14 wheels for heavy plotting
stacks are not guaranteed. Pillow is already a project dependency and can render
charts + assemble a multi-page PDF natively, so the effectiveness report builds
anywhere the rest of the pipeline runs.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

# A4 portrait @ ~150 DPI
PAGE_W, PAGE_H = 1240, 1754
MARGIN = 90
CONTENT_W = PAGE_W - 2 * MARGIN

# Palette (clinical blue/teal/amber)
INK = (33, 37, 41)
MUTED = (108, 117, 125)
ACCENT = (13, 110, 168)
ACCENT2 = (32, 162, 156)
WARN = (214, 137, 16)
BAD = (197, 48, 48)
GOOD = (46, 139, 87)
GRID = (222, 226, 230)
BG = (255, 255, 255)
SERIES = [(13, 110, 168), (32, 162, 156), (214, 137, 16), (140, 94, 191)]

_FONT_CANDIDATES = {
    True: ["arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"],
    False: ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"],
}
_FONT_DIRS = [r"C:\Windows\Fonts", "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"]


class ReportBuilder:
    def __init__(self, title: str, subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle
        self.pages: List[Image.Image] = []
        self._font_cache: Dict[Tuple[int, bool], ImageFont.FreeTypeFont] = {}
        self.page: Optional[Image.Image] = None
        self.draw: Optional[ImageDraw.ImageDraw] = None
        self.y = 0
        self._page_no = 0
        self.new_page(first=True)

    # ----------------------------------------------------------- fonts/util
    def font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        key = (size, bold)
        if key in self._font_cache:
            return self._font_cache[key]
        font = None
        for name in _FONT_CANDIDATES[bold]:
            for d in _FONT_DIRS:
                path = os.path.join(d, name)
                if os.path.exists(path):
                    try:
                        font = ImageFont.truetype(path, size)
                        break
                    except OSError:
                        continue
            if font:
                break
        if font is None:
            font = ImageFont.load_default()
        self._font_cache[key] = font
        return font

    def _text_w(self, text: str, font: ImageFont.FreeTypeFont) -> float:
        return self.draw.textlength(text, font=font)

    def _wrap(self, text: str, font: ImageFont.FreeTypeFont, width: int) -> List[str]:
        words = text.split()
        lines: List[str] = []
        cur = ""
        for w in words:
            trial = (cur + " " + w).strip()
            if self._text_w(trial, font) <= width:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines or [""]

    # ----------------------------------------------------------- pagination
    def new_page(self, first: bool = False) -> None:
        if self.page is not None:
            self._footer()
            self.pages.append(self.page)
        self._page_no += 1
        self.page = Image.new("RGB", (PAGE_W, PAGE_H), BG)
        self.draw = ImageDraw.Draw(self.page)
        # running header
        self.draw.rectangle([0, 0, PAGE_W, 6], fill=ACCENT)
        if not first:
            self.draw.text((MARGIN, 30), self.title, font=self.font(16, True), fill=MUTED)
            self.draw.line([MARGIN, 58, PAGE_W - MARGIN, 58], fill=GRID, width=1)
            self.y = 80
        else:
            self.y = 60

    def _footer(self) -> None:
        f = self.font(14)
        self.draw.line([MARGIN, PAGE_H - 60, PAGE_W - MARGIN, PAGE_H - 60], fill=GRID, width=1)
        self.draw.text((MARGIN, PAGE_H - 50), "OASIS Agentic Pipeline — Effectiveness Analysis",
                       font=f, fill=MUTED)
        num = f"Page {self._page_no}"
        self.draw.text((PAGE_W - MARGIN - self._text_w(num, f), PAGE_H - 50), num, font=f, fill=MUTED)

    def ensure(self, h: int) -> None:
        if self.y + h > PAGE_H - 80:
            self.new_page()

    # ----------------------------------------------------------- primitives
    def space(self, h: int = 16) -> None:
        self.y += h

    def heading(self, text: str, level: int = 1) -> None:
        sizes = {1: 34, 2: 26, 3: 21}
        color = {1: ACCENT, 2: INK, 3: ACCENT2}[level]
        self.ensure(sizes[level] + 26)
        if level == 1:
            self.space(6)
        self.draw.text((MARGIN, self.y), text, font=self.font(sizes[level], True), fill=color)
        self.y += sizes[level] + 8
        if level == 1:
            self.draw.line([MARGIN, self.y, PAGE_W - MARGIN, self.y], fill=ACCENT, width=2)
            self.y += 14

    def paragraph(self, text: str, color=INK, size: int = 19) -> None:
        font = self.font(size)
        for line in self._wrap(text, font, CONTENT_W):
            self.ensure(size + 8)
            self.draw.text((MARGIN, self.y), line, font=font, fill=color)
            self.y += size + 9

    def bullet(self, text: str, color=INK, size: int = 19) -> None:
        font = self.font(size)
        indent = 34
        lines = self._wrap(text, font, CONTENT_W - indent)
        self.ensure(size + 8)
        self.draw.ellipse([MARGIN + 6, self.y + 8, MARGIN + 14, self.y + 16], fill=ACCENT)
        for i, line in enumerate(lines):
            self.ensure(size + 8)
            self.draw.text((MARGIN + indent, self.y), line, font=font, fill=color)
            self.y += size + 9

    def kpi_row(self, items: Sequence[Tuple[str, str, tuple]]) -> None:
        """Render a row of big-number KPI cards: list of (label, value, color)."""
        n = len(items)
        gap = 24
        w = (CONTENT_W - gap * (n - 1)) // n
        h = 150
        self.ensure(h + 20)
        x = MARGIN
        for label, value, color in items:
            self.draw.rounded_rectangle([x, self.y, x + w, self.y + h], radius=14,
                                        fill=(248, 249, 250), outline=GRID, width=1)
            self.draw.rounded_rectangle([x, self.y, x + 8, self.y + h], radius=4, fill=color)
            vfont = self.font(40, True)
            self.draw.text((x + 26, self.y + 30), value, font=vfont, fill=color)
            lfont = self.font(16)
            for j, line in enumerate(self._wrap(label, lfont, w - 40)):
                self.draw.text((x + 26, self.y + 92 + j * 20), line, font=lfont, fill=MUTED)
            x += w + gap
        self.y += h + 20

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[str]],
              col_w: Optional[Sequence[int]] = None) -> None:
        ncol = len(headers)
        if col_w is None:
            col_w = [CONTENT_W // ncol] * ncol
        rh = 38
        hf = self.font(17, True)
        bf = self.font(17)
        self.ensure(rh * 2)
        # header
        x = MARGIN
        self.draw.rectangle([MARGIN, self.y, MARGIN + sum(col_w), self.y + rh], fill=ACCENT)
        for h, w in zip(headers, col_w):
            self.draw.text((x + 12, self.y + 9), str(h), font=hf, fill=(255, 255, 255))
            x += w
        self.y += rh
        # rows
        for ri, row in enumerate(rows):
            self.ensure(rh)
            bg = (255, 255, 255) if ri % 2 == 0 else (245, 247, 249)
            self.draw.rectangle([MARGIN, self.y, MARGIN + sum(col_w), self.y + rh], fill=bg)
            x = MARGIN
            for c, w in zip(row, col_w):
                self.draw.text((x + 12, self.y + 9), str(c), font=bf, fill=INK)
                x += w
            self.y += rh
        self.draw.rectangle([MARGIN, self.y - rh * (len(rows) + 1), MARGIN + sum(col_w), self.y],
                            outline=GRID, width=1)
        self.y += 10

    # ----------------------------------------------------------- charts
    def _chart_box(self, h: int) -> Tuple[int, int, int, int]:
        self.ensure(h + 20)
        x0, y0 = MARGIN, self.y
        return x0, y0, CONTENT_W, h

    def grouped_bar(self, title: str, categories: Sequence[str],
                    series: Dict[str, Sequence[float]], ymax: float = 1.0,
                    height: int = 360, fmt: str = "{:.2f}") -> None:
        self.heading(title, level=3)
        x0, y0, w, h = self._chart_box(height)
        pad_l, pad_b, pad_t = 60, 70, 20
        plot_w = w - pad_l - 10
        plot_h = h - pad_b - pad_t
        ax_x = x0 + pad_l
        ax_y = y0 + pad_t
        # gridlines + y labels
        for i in range(5):
            gy = ax_y + plot_h - int(plot_h * i / 4)
            self.draw.line([ax_x, gy, ax_x + plot_w, gy], fill=GRID, width=1)
            val = ymax * i / 4
            self.draw.text((x0, gy - 9), f"{val:.1f}", font=self.font(14), fill=MUTED)
        names = list(series.keys())
        ncat = len(categories)
        nser = len(names)
        group_w = plot_w / ncat
        bar_w = group_w * 0.7 / max(nser, 1)
        for ci, cat in enumerate(categories):
            gx = ax_x + ci * group_w + group_w * 0.15
            for si, name in enumerate(names):
                val = max(0.0, min(series[name][ci], ymax))
                bh = int(plot_h * val / ymax) if ymax else 0
                bx = gx + si * bar_w
                color = SERIES[si % len(SERIES)]
                self.draw.rectangle([bx, ax_y + plot_h - bh, bx + bar_w - 4, ax_y + plot_h], fill=color)
                lbl = fmt.format(series[name][ci])
                lf = self.font(13)
                self.draw.text((bx, ax_y + plot_h - bh - 18), lbl, font=lf, fill=INK)
            # category label
            cf = self.font(14)
            for j, line in enumerate(self._wrap(cat, cf, int(group_w))):
                self.draw.text((ax_x + ci * group_w + 6, ax_y + plot_h + 8 + j * 16), line,
                               font=cf, fill=INK)
        # axes
        self.draw.line([ax_x, ax_y, ax_x, ax_y + plot_h], fill=INK, width=2)
        self.draw.line([ax_x, ax_y + plot_h, ax_x + plot_w, ax_y + plot_h], fill=INK, width=2)
        # legend
        lx = ax_x
        ly = y0 + h - 24
        for si, name in enumerate(names):
            color = SERIES[si % len(SERIES)]
            self.draw.rectangle([lx, ly, lx + 18, ly + 14], fill=color)
            self.draw.text((lx + 24, ly - 2), name, font=self.font(14), fill=INK)
            lx += 40 + int(self._text_w(name, self.font(14)))
        self.y = y0 + h + 16

    def heatmap(self, title: str, matrix: Sequence[Sequence[float]],
                row_labels: Sequence[str], col_labels: Sequence[str],
                height: int = 460, annotate_norm: bool = True) -> None:
        self.heading(title, level=3)
        n = len(matrix)
        m = len(matrix[0]) if n else 0
        x0, y0, w, h = self._chart_box(height)
        pad_l, pad_t, pad_b = 150, 30, 110
        cell = min((w - pad_l - 20) // max(m, 1), (h - pad_t - pad_b) // max(n, 1))
        grid_x = x0 + pad_l
        grid_y = y0 + pad_t
        mx = max((max(r) for r in matrix), default=1) or 1
        for i in range(n):
            row_total = sum(matrix[i]) or 1
            for j in range(m):
                v = matrix[i][j]
                t = v / mx
                # white -> accent blue
                col = (int(255 - t * (255 - ACCENT[0])),
                       int(255 - t * (255 - ACCENT[1])),
                       int(255 - t * (255 - ACCENT[2])))
                cx0 = grid_x + j * cell
                cy0 = grid_y + i * cell
                self.draw.rectangle([cx0, cy0, cx0 + cell, cy0 + cell], fill=col, outline=GRID)
                txt_col = (255, 255, 255) if t > 0.55 else INK
                vf = self.font(16, True)
                vs = str(int(v))
                self.draw.text((cx0 + cell / 2 - self._text_w(vs, vf) / 2, cy0 + cell / 2 - 16),
                               vs, font=vf, fill=txt_col)
                if annotate_norm:
                    pf = self.font(12)
                    ps = f"{100*v/row_total:.0f}%"
                    self.draw.text((cx0 + cell / 2 - self._text_w(ps, pf) / 2, cy0 + cell / 2 + 4),
                                   ps, font=pf, fill=txt_col)
        # row labels (true)
        rf = self.font(14)
        for i, lab in enumerate(row_labels):
            for k, line in enumerate(self._wrap(lab, rf, pad_l - 12)):
                self.draw.text((x0, grid_y + i * cell + cell / 2 - 8 + k * 15), line, font=rf, fill=INK)
        # col labels (pred)
        for j, lab in enumerate(col_labels):
            cf = self.font(14)
            for k, line in enumerate(self._wrap(lab, cf, cell + 10)):
                self.draw.text((grid_x + j * cell + 4, grid_y + n * cell + 8 + k * 15),
                               line, font=cf, fill=INK)
        self.draw.text((x0, grid_y + n * cell + 70), "Rows = true class   Columns = predicted class   "
                       "(cell = count, % = row-normalized recall)", font=self.font(13), fill=MUTED)
        self.y = y0 + h + 16

    def hbar(self, title: str, labels: Sequence[str], values: Sequence[float],
             colors: Optional[Sequence[tuple]] = None, height: int = 320,
             unit: str = "", vmax: Optional[float] = None) -> None:
        self.heading(title, level=3)
        x0, y0, w, h = self._chart_box(height)
        pad_l = 230
        n = len(labels)
        bar_area = w - pad_l - 120
        rowh = (h - 20) / max(n, 1)
        vmax = vmax if vmax is not None else (max(values) if values else 1) or 1
        for i, (lab, val) in enumerate(zip(labels, values)):
            cy = y0 + i * rowh + rowh * 0.2
            bh = rowh * 0.55
            color = colors[i] if colors else SERIES[i % len(SERIES)]
            bw = int(bar_area * min(val / vmax, 1.0))
            self.draw.text((x0, cy + bh / 2 - 10), lab, font=self.font(15), fill=INK)
            self.draw.rectangle([x0 + pad_l, cy, x0 + pad_l + bw, cy + bh], fill=color)
            self.draw.text((x0 + pad_l + bw + 10, cy + bh / 2 - 10),
                           f"{val:.2f}{unit}", font=self.font(15, True), fill=INK)
        self.y = y0 + h + 16

    # ----------------------------------------------------------- output
    def cover(self, meta_lines: Sequence[str]) -> None:
        d = self.draw
        d.rectangle([0, 0, PAGE_W, 320], fill=ACCENT)
        d.rectangle([0, 320, PAGE_W, 332], fill=ACCENT2)
        d.text((MARGIN, 120), "OASIS Agentic Pipeline", font=self.font(52, True), fill=(255, 255, 255))
        d.text((MARGIN, 195), "Effectiveness & Data Analysis Report", font=self.font(30),
               fill=(230, 240, 250))
        self.y = 400
        self.paragraph(self.subtitle, color=MUTED, size=20)
        self.space(20)
        for line in meta_lines:
            self.bullet(line, size=18)
        self.y = PAGE_H - 230
        self.draw.line([MARGIN, self.y, PAGE_W - MARGIN, self.y], fill=GRID, width=1)
        self.space(16)
        self.paragraph("Hybrid edge-cloud, Snapdragon-NPU-optimized, zero paid-API multi-agent "
                       "system for Alzheimer's disease screening on the OASIS datasets.",
                       color=MUTED, size=17)

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # finalize last page
        self._footer()
        self.pages.append(self.page)
        self.page = None
        first, rest = self.pages[0], self.pages[1:]
        first.save(path, "PDF", save_all=True, append_images=rest, resolution=150.0)
        return path
