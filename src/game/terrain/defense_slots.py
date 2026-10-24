from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pygame


@dataclass(slots=True)
class DefenseSlot:
    """A rectangular location where a defense element can be placed."""

    slot_id: int
    rect: pygame.Rect
    occupied: bool = False

    @property
    def center(self) -> pygame.Vector2:
        """Return the center of the slot in world coordinates."""
        return pygame.Vector2(self.rect.center)


class DefenseSlots:
    """Load and render rectangular placement slots from a Tiled map.

    The requested Tiled object layer is configurable. This allows the same
    loader to manage:
        - defense_slots
        - resource_slots
        - king_tower_slots

    Slot coordinates are expressed in map/world pixels. Rendering scales
    those coordinates to the supplied surface, preserving the original
    rendering behavior.
    """

    def __init__(
        self,
        map_path: Path,
        slot_image_path: Path | None = None,
        *,
        layer_name: str = "defense_slots",
        load_image: bool = True,
    ) -> None:
        map_path = Path(map_path).resolve()

        if not map_path.is_file():
            raise FileNotFoundError(f"Slot map not found: {map_path}")

        if not isinstance(layer_name, str) or not layer_name.strip():
            raise ValueError("Layer name must be a non-empty string.")

        self.layer_name = layer_name.strip()

        with map_path.open("r", encoding="utf-8") as file:
            map_data = json.load(file)

        try:
            map_width_tiles = int(map_data["width"])
            map_height_tiles = int(map_data["height"])
            self.tile_width = int(map_data["tilewidth"])
            self.tile_height = int(map_data["tileheight"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Tiled map dimensions in {map_path}.") from exc

        if (
            map_width_tiles <= 0
            or map_height_tiles <= 0
            or self.tile_width <= 0
            or self.tile_height <= 0
        ):
            raise ValueError("Map dimensions and tile dimensions must be positive.")

        self.map_width = map_width_tiles * self.tile_width
        self.map_height = map_height_tiles * self.tile_height

        self.slot_image: pygame.Surface | None = None

        if load_image and slot_image_path is not None:
            slot_image_path = Path(slot_image_path).resolve()

            if not slot_image_path.is_file():
                raise FileNotFoundError(f"Slot image not found: {slot_image_path}")

            self.slot_image = pygame.image.load(str(slot_image_path)).convert_alpha()

        self.slots: list[DefenseSlot] = []
        self._load_slots(map_data.get("layers", []))

    @staticmethod
    def _iter_layers(
        layers: list[dict[str, Any]],
    ):
        """Yield object and other layers, including nested group layers."""
        for layer in layers:
            yield layer

            if layer.get("type") == "group":
                yield from DefenseSlots._iter_layers(layer.get("layers", []))

    def _load_slots(self, layers: list[dict[str, Any]]) -> None:
        """Load rectangular slot objects from the configured Tiled layer."""
        normalized_name = self.layer_name.casefold()

        slot_layer = next(
            (
                layer
                for layer in self._iter_layers(layers)
                if layer.get("type") == "objectgroup"
                and str(layer.get("name", "")).strip().casefold() == normalized_name
            ),
            None,
        )

        if slot_layer is None:
            available_layers = [
                str(layer.get("name", ""))
                for layer in self._iter_layers(layers)
                if layer.get("type") == "objectgroup"
            ]

            raise ValueError(
                f"No Tiled object layer named '{self.layer_name}' "
                f"was found. Available object layers: {available_layers}"
            )

        # Tiled layer x/y are expressed in tile units; offsetx/offsety
        # are pixel offsets.
        layer_x = float(slot_layer.get("x", 0)) * self.tile_width + float(
            slot_layer.get("offsetx", 0)
        )
        layer_y = float(slot_layer.get("y", 0)) * self.tile_height + float(
            slot_layer.get("offsety", 0)
        )

        loaded_slots: list[DefenseSlot] = []
        seen_ids: set[int] = set()

        for obj in slot_layer.get("objects", []):
            width = float(obj.get("width", 0))
            height = float(obj.get("height", 0))

            # Ignore point objects and objects without a positive area.
            if width <= 0 or height <= 0:
                continue

            rotation = float(obj.get("rotation", 0))

            if not math.isfinite(rotation):
                raise ValueError(
                    f"Slot object {obj.get('id')} has an invalid rotation."
                )

            if not math.isclose(rotation, 0.0, abs_tol=1e-6):
                raise ValueError(
                    f"Slot object {obj.get('id')} in layer "
                    f"'{self.layer_name}' is rotated. "
                    "Only unrotated rectangles are supported."
                )

            try:
                slot_id = int(obj["id"])
                x = float(obj.get("x", 0)) + layer_x
                y = float(obj.get("y", 0)) + layer_y
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid slot object in layer '{self.layer_name}'."
                ) from exc

            if slot_id in seen_ids:
                raise ValueError(
                    f"Duplicate slot ID {slot_id} in layer '{self.layer_name}'."
                )

            if not all(math.isfinite(value) for value in (x, y, width, height)):
                raise ValueError(f"Slot object {slot_id} contains non-finite geometry.")

            rect = pygame.Rect(
                round(x),
                round(y),
                round(width),
                round(height),
            )

            if rect.width <= 0 or rect.height <= 0:
                raise ValueError(
                    f"Slot object {slot_id} has no usable area after "
                    "converting its dimensions to pixels."
                )

            loaded_slots.append(
                DefenseSlot(
                    slot_id=slot_id,
                    rect=rect,
                )
            )
            seen_ids.add(slot_id)

        loaded_slots.sort(key=lambda slot: slot.slot_id)

        if not loaded_slots:
            raise ValueError(
                f"No rectangular slots were found in Tiled layer '{self.layer_name}'."
            )

        self.slots = loaded_slots

    def draw(
        self,
        surface: pygame.Surface,
        show_ids: bool = False,
    ) -> None:
        """Draw icons for unoccupied slots, optionally showing their IDs."""
        if self.slot_image is None:
            raise RuntimeError(
                f"Slot images were not loaded for layer '{self.layer_name}'."
            )

        if all(slot.occupied for slot in self.slots):
            return

        scale_x = surface.get_width() / self.map_width
        scale_y = surface.get_height() / self.map_height

        overlay = pygame.Surface(
            surface.get_size(),
            pygame.SRCALPHA,
        )

        font = pygame.font.Font(None, 18) if show_ids else None

        for slot in self.slots:
            if slot.occupied:
                continue

            screen_rect = pygame.Rect(
                round(slot.rect.x * scale_x),
                round(slot.rect.y * scale_y),
                max(1, round(slot.rect.width * scale_x)),
                max(1, round(slot.rect.height * scale_y)),
            )

            icon = pygame.transform.smoothscale(
                self.slot_image,
                screen_rect.size,
            )
            overlay.blit(icon, screen_rect)

            if font is not None:
                label = font.render(
                    str(slot.slot_id),
                    True,
                    (255, 255, 255),
                )
                label_rect = label.get_rect(
                    center=screen_rect.center,
                )
                background_rect = label_rect.inflate(6, 4)

                pygame.draw.rect(
                    overlay,
                    (20, 25, 25, 200),
                    background_rect,
                    border_radius=3,
                )
                overlay.blit(label, label_rect)

        surface.blit(overlay, (0, 0))
