"""
Shared multimodal (image / file) helpers for provider adapters.

Normalises SDK intermediate content parts (``TextPart`` / ``ImagePart`` /
``FilePart`` dicts from :mod:`ai_sdk.types`) into provider-native shapes and
collects image/file payloads from provider responses into the common
``files`` list used by :func:`ai_sdk.generate_text`.
"""

from __future__ import annotations

import base64
import re
from typing import Any

_DATA_URI_RE = re.compile(
    r"^data:(?P<mime>[^;,]+)?(?:;[^,]*)?;base64,(?P<data>.+)$",
    re.DOTALL,
)


def decode_data_uri(value: str) -> tuple[bytes, str | None]:
    """Decode a ``data:`` URI into ``(raw_bytes, mime_type_or_none)``."""
    match = _DATA_URI_RE.match(value.strip())
    if not match:
        raise ValueError("Not a base64 data URI")
    mime = match.group("mime") or None
    return base64.b64decode(match.group("data")), mime


def coerce_image_bytes(
    image: str | bytes,
    *,
    mime_type: str | None = None,
) -> tuple[bytes, str]:
    """Return ``(bytes, mime_type)`` for an image part value."""
    if isinstance(image, bytes):
        return image, mime_type or "image/png"
    if isinstance(image, str) and image.startswith("data:"):
        data, mime_from_uri = decode_data_uri(image)
        return data, mime_from_uri or mime_type or "image/png"
    if isinstance(image, str):
        try:
            return base64.b64decode(image, validate=True), mime_type or "image/png"
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                "Image string must be a data: URI, base64 payload, or pass bytes"
            ) from exc
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def image_to_data_uri(image: str | bytes, *, mime_type: str | None = None) -> str:
    """Build a ``data:<mime>;base64,...`` URI or pass through an existing one."""
    if isinstance(image, str) and image.startswith("data:"):
        return image
    data, mime = coerce_image_bytes(image, mime_type=mime_type)
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def image_to_base64(
    image: str | bytes, *, mime_type: str | None = None
) -> tuple[str, str]:
    """Return ``(base64_str, mime_type)`` without a data-URI wrapper."""
    if isinstance(image, str) and image.startswith("data:"):
        data, mime = decode_data_uri(image)
        return base64.b64encode(data).decode("ascii"), mime or mime_type or "image/png"
    data, mime = coerce_image_bytes(image, mime_type=mime_type)
    return base64.b64encode(data).decode("ascii"), mime


def part_mime(part: dict[str, Any], default: str = "application/octet-stream") -> str:
    return part.get("mimeType") or part.get("mime_type") or default


# ---------------------------------------------------------------------------
# OpenAI Chat Completions content parts
# ---------------------------------------------------------------------------


def openai_content_part_from_sdk(part: Any) -> dict[str, Any] | None:
    """Map one SDK content part (dict or model dump) to an OpenAI content part."""
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        if hasattr(part, "to_dict"):
            part = part.to_dict()
        else:
            return {"type": "text", "text": str(part)}

    ptype = part.get("type")
    if ptype == "text" or (ptype is None and "text" in part and "image" not in part):
        return {"type": "text", "text": part.get("text", "")}

    if ptype == "image" or "image" in part:
        image_val = part.get("image")
        mime = part_mime(part, "image/png")
        if isinstance(image_val, str) and (
            image_val.startswith("http://") or image_val.startswith("https://")
        ):
            return {"type": "image_url", "image_url": {"url": image_val}}
        url = image_to_data_uri(image_val, mime_type=mime)
        return {"type": "image_url", "image_url": {"url": url}}

    if ptype == "file":
        data_val = part.get("data")
        mime = part_mime(part)
        if mime.startswith("image/"):
            url = image_to_data_uri(data_val, mime_type=mime)
            return {"type": "image_url", "image_url": {"url": url}}
        if isinstance(data_val, bytes):
            b64 = base64.b64encode(data_val).decode("ascii")
        elif isinstance(data_val, str) and data_val.startswith("data:"):
            b64, mime = image_to_base64(data_val, mime_type=mime)
        else:
            b64 = str(data_val)
        preview = b64 if len(b64) <= 120 else f"{b64[:120]}…"
        return {"type": "text", "text": f"[file mime={mime} base64={preview}]"}

    return {"type": "text", "text": str(part)}


def normalise_openai_message_content(content: Any) -> Any:
    """Normalise user/assistant ``content`` for OpenAI (string or multi-part list)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [openai_content_part_from_sdk(p) for p in content]
        parts = [p for p in parts if p is not None]
        if not parts:
            return ""
        if len(parts) == 1 and parts[0].get("type") == "text":
            return parts[0].get("text", "")
        return parts
    return str(content)


def files_from_openai_message(message: Any) -> list[dict[str, Any]]:
    """Extract image/file payloads from an OpenAI chat completion message."""
    files: list[dict[str, Any]] = []
    content = getattr(message, "content", None)
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "image_url":
                    url = (item.get("image_url") or {}).get("url")
                    if isinstance(url, str) and url.startswith("data:"):
                        try:
                            data, mime = decode_data_uri(url)
                            files.append(
                                {
                                    "base64": base64.b64encode(data).decode("ascii"),
                                    "uint8_array": data,
                                    "mime_type": mime,
                                }
                            )
                        except ValueError:
                            pass
                continue
            itype = getattr(item, "type", None)
            if itype == "image_url":
                image_url = getattr(item, "image_url", None)
                url = (
                    image_url.get("url")
                    if isinstance(image_url, dict)
                    else getattr(image_url, "url", None)
                )
                if isinstance(url, str) and url.startswith("data:"):
                    try:
                        data, mime = decode_data_uri(url)
                        files.append(
                            {
                                "base64": base64.b64encode(data).decode("ascii"),
                                "uint8_array": data,
                                "mime_type": mime,
                            }
                        )
                    except ValueError:
                        pass

    images = getattr(message, "images", None) or []
    for img in images:
        b64 = getattr(img, "b64_json", None) or (
            img.get("b64_json") if isinstance(img, dict) else None
        )
        if b64:
            try:
                data = base64.b64decode(b64)
            except Exception:  # noqa: BLE001
                data = None
            files.append(
                {
                    "base64": b64 if isinstance(b64, str) else None,
                    "uint8_array": data,
                    "mime_type": "image/png",
                }
            )
    return files


# ---------------------------------------------------------------------------
# Anthropic Messages API content blocks
# ---------------------------------------------------------------------------


def anthropic_block_from_sdk_part(part: Any) -> dict[str, Any] | None:
    """Map one SDK content part to an Anthropic content block."""
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        if hasattr(part, "to_dict"):
            part = part.to_dict()
        else:
            return {"type": "text", "text": str(part)}

    ptype = part.get("type")
    if ptype == "text" or (ptype is None and "text" in part and "image" not in part):
        return {"type": "text", "text": part.get("text", "")}

    if ptype == "image" or "image" in part:
        image_val = part.get("image")
        mime = part_mime(part, "image/png")
        if isinstance(image_val, str) and (
            image_val.startswith("http://") or image_val.startswith("https://")
        ):
            return {"type": "image", "source": {"type": "url", "url": image_val}}
        b64, mime = image_to_base64(image_val, mime_type=mime)
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": b64},
        }

    if ptype == "file":
        data_val = part.get("data")
        mime = part_mime(part)
        if mime.startswith("image/"):
            b64, mime = image_to_base64(data_val, mime_type=mime)
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64},
            }
        if isinstance(data_val, bytes):
            b64 = base64.b64encode(data_val).decode("ascii")
        elif isinstance(data_val, str) and data_val.startswith("data:"):
            b64, mime = image_to_base64(data_val, mime_type=mime)
        else:
            b64 = str(data_val)
        if mime == "application/pdf" or mime.endswith("/pdf"):
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": b64,
                },
            }
        return {"type": "text", "text": f"[attached file mime={mime}]"}

    if ptype == "tool-call":
        return {
            "type": "tool_use",
            "id": part.get("toolCallId") or part.get("tool_call_id", "tool-call"),
            "name": part.get("toolName") or part.get("tool_name", "tool"),
            "input": part.get("args", {}),
        }

    return {"type": "text", "text": str(part)}


def normalise_anthropic_user_content(content: Any) -> Any:
    """Return Anthropic ``content`` (string or list of blocks) for a user message."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        blocks = [anthropic_block_from_sdk_part(p) for p in content]
        blocks = [b for b in blocks if b is not None]
        return blocks if blocks else ""
    return str(content)


def files_from_anthropic_message(resp: Any) -> list[dict[str, Any]]:
    """Collect image/document blocks from an Anthropic message response."""
    files: list[dict[str, Any]] = []
    for block in getattr(resp, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type not in ("image", "document"):
            continue
        source = getattr(block, "source", None)
        if source is None:
            continue
        data_b64 = getattr(source, "data", None) or (
            source.get("data") if isinstance(source, dict) else None
        )
        mime = getattr(source, "media_type", None) or (
            source.get("media_type") if isinstance(source, dict) else None
        )
        if block_type == "document" and not mime:
            mime = "application/pdf"
        if not data_b64:
            continue
        try:
            data = base64.b64decode(data_b64)
        except Exception:  # noqa: BLE001
            data = None
        files.append(
            {
                "base64": data_b64 if isinstance(data_b64, str) else None,
                "uint8_array": data,
                "mime_type": mime,
            }
        )
    return files


# ---------------------------------------------------------------------------
# Gemini descriptors (caller builds google.genai.types.Part)
# ---------------------------------------------------------------------------


def gemini_part_descriptors_from_sdk_content(content: Any) -> list[dict[str, Any]]:
    """Build descriptors: ``text`` / ``bytes``+``mime_type`` / ``uri``+``mime_type``."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return [{"text": str(content)}]

    out: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            out.append({"text": item})
            continue
        if not isinstance(item, dict):
            if hasattr(item, "to_dict"):
                item = item.to_dict()
            else:
                out.append({"text": str(item)})
                continue

        ptype = item.get("type")
        is_text = ptype == "text" or (
            ptype is None and "text" in item and "image" not in item
        )
        if is_text:
            out.append({"text": item.get("text", "")})
        elif ptype == "image" or "image" in item:
            image_val = item.get("image")
            mime = part_mime(item, "image/png")
            if isinstance(image_val, str) and (
                image_val.startswith("http://") or image_val.startswith("https://")
            ):
                out.append({"uri": image_val, "mime_type": mime})
            else:
                data, mime = coerce_image_bytes(image_val, mime_type=mime)
                out.append({"bytes": data, "mime_type": mime})
        elif ptype == "file":
            data_val = item.get("data")
            mime = part_mime(item)
            if isinstance(data_val, str) and (
                data_val.startswith("http://") or data_val.startswith("https://")
            ):
                out.append({"uri": data_val, "mime_type": mime})
            elif isinstance(data_val, bytes):
                out.append({"bytes": data_val, "mime_type": mime})
            elif isinstance(data_val, str) and data_val.startswith("data:"):
                data, mime_from_uri = decode_data_uri(data_val)
                out.append({"bytes": data, "mime_type": mime_from_uri or mime})
            else:
                out.append({"text": f"[file mime={mime}]"})
        else:
            out.append({"text": str(item)})
    return out


def files_from_gemini_response(resp: Any) -> list[dict[str, Any]]:
    """Collect inline image/file parts from a Gemini generateContent response."""
    files: list[dict[str, Any]] = []
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        return files
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline is None:
            continue
        data = getattr(inline, "data", None)
        mime = getattr(inline, "mime_type", None) or getattr(inline, "mimeType", None)
        if data is None:
            continue
        if isinstance(data, str):
            try:
                raw = base64.b64decode(data)
                b64 = data
            except Exception:  # noqa: BLE001
                raw = data.encode("utf-8")
                b64 = base64.b64encode(raw).decode("ascii")
        else:
            raw = data
            b64 = base64.b64encode(bytes(data)).decode("ascii")
        files.append(
            {
                "base64": b64,
                "uint8_array": bytes(raw) if not isinstance(raw, bytes) else raw,
                "mime_type": mime,
            }
        )
    return files
