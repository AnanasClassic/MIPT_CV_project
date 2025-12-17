"""
Generate a dataset of rendered integral images from a CSV of integral templates.

Output structure:
  OUT_DIR/
    metadata.csv
    images/
      img_0000000.png
      img_0000001.png
      ...

metadata.csv format:
tex,image_path
"\\int_{-2}^{3} ... \\,dx",images/img_0000000.png
...

Usage:
  python make_integral_dataset.py --num 1000 --out ./out --templates ./integrals.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class IntegralTemplate:
    section: str
    lhs: str
    rhs: str
    params: List[str]


class IntegralGenerator:
    """
    Picks a random LHS from templates and substitutes parameters:
      - a,b,c: randint(-9..9 excluding 0) OR '\\pi' OR 'e' OR itself ('a'/'b'/'c')
      - n: randint(2..9) OR '\\pi' OR 'e' OR 'n'
    Also normalizes "+ -k" -> "-k", "- -k" -> "+k", etc.
    """

    def __init__(self, templates: List[IntegralTemplate], seed: Optional[int] = None):
        self.templates = templates
        self.rng = random.Random(seed)

    @staticmethod
    def from_csv(path: Union[str, Path], encoding: str = "utf-8") -> "IntegralGenerator":
        path = Path(path)
        with path.open("r", encoding=encoding, newline="") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            if not reader.fieldnames:
                raise ValueError("CSV has no header.")
            reader.fieldnames = [fn.strip() for fn in reader.fieldnames]

            templates: List[IntegralTemplate] = []
            for row in reader:
                row = {k.strip(): (v or "") for k, v in row.items()}

                section = row.get("Section", "")
                lhs = row.get("Integral (LHS)", "")
                rhs = row.get("Result (RHS)", "")
                params_raw = row.get("parameter", "").strip()

                params = [p for p in re.split(r"[,\s]+", params_raw) if p]
                templates.append(IntegralTemplate(section=section, lhs=lhs, rhs=rhs, params=params))

        if not templates:
            raise ValueError("No templates loaded from CSV.")
        return IntegralGenerator(templates)

    def _choice_for_param(self, p: str) -> str:
        if p in ("a", "b", "c"):
            choices = [str(i) for i in range(-9, 10) if i != 0] + [r"\pi ", "e", p]
            return self.rng.choice(choices)
        if p == "n":
            choices = [str(i) for i in range(2, 10)] + [r"\pi ", "e", "n"]
            return self.rng.choice(choices)
        return p

    @staticmethod
    def _last_nonspace_char(out: List[str]) -> str:
        for s in reversed(out):
            if not s:
                continue
            j = len(s) - 1
            while j >= 0 and s[j].isspace():
                j -= 1
            if j >= 0:
                return s[j]
        return ""

    @classmethod
    def _normalize_signs(cls, s: str) -> str:
        s = re.sub(r"\+\s*-", "-", s)
        s = re.sub(r"-\s*-", "+", s)
        s = re.sub(r"\+\s*\+", "+", s)

        s = re.sub(r"\s{2,}", " ", s)
        return s

    @classmethod
    def _substitute_latex(cls, expr: str, mapping: Dict[str, str]) -> str:
        """
        Substitution that:
          - does NOT replace inside LaTeX command names (after backslash),
          - wraps negative ints in exponent/base contexts,
          - inserts '\\cdot ' before 'e' if it would stick to a preceding digit (2e -> 2\\cdot e),
          - normalizes duplicate sign artifacts.
        """
        out: List[str] = []
        i = 0

        while i < len(expr):
            ch = expr[i]

            if ch == "\\":
                out.append(ch)
                i += 1
                while i < len(expr) and expr[i].isalpha():
                    out.append(expr[i])
                    i += 1
                continue

            if ch in mapping:
                rep = mapping[ch]

                prev_ch_expr = expr[i - 1] if i - 1 >= 0 else ""
                next_ch_expr = expr[i + 1] if i + 1 < len(expr) else ""

                if rep == "e" and prev_ch_expr != "^":
                    last = cls._last_nonspace_char(out)
                    if last.isdigit():
                        out.append(r"\cdot ")

                is_neg_int = rep.startswith("-") and rep[1:].isdigit()

                if is_neg_int and prev_ch_expr == "^":
                    out.append("{" + rep + "}")
                elif is_neg_int and next_ch_expr == "^":
                    out.append("(" + rep + ")")
                else:
                    out.append(rep)
            else:
                out.append(ch)

            i += 1

        s = "".join(out)
        s = cls._normalize_signs(s)
        return s

    def generate_random_integral_tex(self) -> str:
        tpl = self.rng.choice(self.templates)
        mapping: Dict[str, str] = {p: self._choice_for_param(p) for p in tpl.params}
        lhs = self._substitute_latex(tpl.lhs, mapping)
        return lhs

    def generate_k(self, k: int) -> List[str]:
        integral_set = set()
        while(len(integral_set) < k):
            integral_set.add(self.generate_random_integral_tex())
        return list(integral_set)


def add_random_bounds_if_missing(tex: str, rng: random.Random, low: int = -3, high: int = 3) -> str:
    """
    If tex contains '\\int' without _{...}^{...}, add random integer bounds:
      \\int f(x) dx  ->  \\int_{-2}^{3} f(x) dx
    """
    if r"\int_" in tex:
        return tex

    vals = list(range(low, high + 1))
    l = rng.choice(vals)
    u = rng.choice(vals)
    tries = 0
    while (u <= l) and tries < 50:
        l = rng.choice(vals)
        u = rng.choice(vals)
        tries += 1
    if u <= l:
        l, u = low, high

    return tex.replace(r"\int", rf"\int_{{{l}}}^{{{u}}}", 1)



def render_integral_to_png(tex: str, out_path: Union[str, Path], dpi: int = 200, fontsize: int = 32, pad_inches: float = 0.15) -> None:
    """
    Renders LaTeX math (matplotlib mathtext) into a PNG file.
    `tex` should NOT include surrounding $...$ (we add them).
    """
    tex = re.sub(r"dx", "\\ dx", tex)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(0.01, 0.01), dpi=dpi)
    fig.patch.set_facecolor("white")

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(
        0.5, 0.5,
        f"${tex}$",
        ha="center", va="center",
        fontsize=fontsize
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches, facecolor="white")
    plt.close(fig)


def create_integral_dataset(
    num_integrals: int,
    out_dir: Union[str, Path],
    templates_csv: Union[str, Path],
    seed: Optional[int] = 0,
    add_bounds: bool = True,
    dpi: int = 200,
    fontsize: int = 32,
) -> Path:
    """
    Creates:
      - out_dir/metadata.csv
      - out_dir/images/img_XXXXXXXX.png
    Returns path to metadata.csv
    """
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    gen = IntegralGenerator.from_csv(templates_csv)
    rng = random.Random(seed)

    meta_path = out_dir / "metadata.csv"
    with meta_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tex", "image_path"])

        integrals = gen.generate_k(num_integrals)
        for i, tex in enumerate(tqdm(integrals, desc="visualizing integrals")):
            if add_bounds:
                tex = add_random_bounds_if_missing(tex, rng=rng, low=-3, high=3)

            tex = IntegralGenerator._normalize_signs(tex)

            rel_img = Path("images") / f"img_{i:07d}.png"
            abs_img = out_dir / rel_img

            render_integral_to_png(tex, abs_img, dpi=dpi, fontsize=fontsize)
            writer.writerow([tex, rel_img.as_posix()])

    return meta_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, required=True, help="Number of integrals to generate")
    p.add_argument("--out", type=str, required=True, help="Output directory")
    p.add_argument("--templates", type=str, default="integrals.csv", help="CSV with templates (default: integrals.csv)")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    p.add_argument("--no-bounds", action="store_false", help="Do not add random bounds to \\int if missing")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--fontsize", type=int, default=32)
    args = p.parse_args()

    meta = create_integral_dataset(
        num_integrals=args.num,
        out_dir=args.out,
        templates_csv=args.templates,
        seed=args.seed,
        add_bounds=not args.no_bounds,
        dpi=args.dpi,
        fontsize=args.fontsize,
    )
    print(str(meta))


if __name__ == "__main__":
    main()
