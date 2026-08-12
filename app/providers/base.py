from __future__ import annotations

from typing import Protocol


class SketchProvider(Protocol):
    name: str

    def available(self) -> bool: ...

    async def sketch(self, design: dict) -> dict:
        """Return {image_base64|image_url, model}."""
        ...


class TryonProvider(Protocol):
    name: str

    def available(self) -> bool: ...

    async def tryon(self, photo: bytes, design: dict) -> dict:
        """Return {image_base64|image_url, model}."""
        ...


def build_prompt(design: dict) -> str:
    """Turn the component tree Rails sends into a text prompt.

    Rails posts structured modules (neck/sleeve/dress_type/fabric/colors/
    embroidery/lace) rather than prose, so the wording is assembled here where
    it can be tuned without a backend deploy.
    """
    neck = design.get("neck") or {}
    colors = design.get("colors") or {}
    embroidery = design.get("embroidery") or {}
    lace = design.get("lace") or {}

    parts: list[str] = ["fashion design sketch of a womens garment"]

    if design.get("dress_type"):
        parts.append(f"{design['dress_type']} silhouette")
    if neck.get("front"):
        parts.append(f"{neck['front']} neckline")
    if design.get("sleeve"):
        parts.append(f"{design['sleeve']}")
    if design.get("fabric"):
        parts.append(f"{design['fabric']} fabric")
    if design.get("pattern") and design["pattern"].lower() != "plain":
        parts.append(f"{design['pattern']} pattern")

    bodice = colors.get("bodice")
    skirt = colors.get("skirt")
    if bodice and skirt and bodice != skirt:
        parts.append(f"bodice in {bodice}, skirt in {skirt}")
    elif bodice or skirt or colors.get("primary"):
        parts.append(f"in {bodice or skirt or colors['primary']}")

    placements = [
        key
        for key in ("neck", "bodice_upper", "bodice_lower", "skirt_upper", "skirt_lower")
        if embroidery.get(key)
    ]
    if placements and embroidery.get("style"):
        parts.append(f"{embroidery['style']} embroidery on the {', '.join(placements)}")

    if lace.get("style"):
        parts.append(f"{lace['style']} lace trim")
    if lace.get("has_belt"):
        parts.append("with a waist belt")

    parts.append(
        "full length front view, flat lay technical fashion illustration, "
        "clean white background, no model, no text"
    )
    return ", ".join(parts)


NEGATIVE_PROMPT = (
    "photograph, human face, mannequin head, text, watermark, logo, "
    "cluttered background, multiple garments, low quality, blurry"
)
