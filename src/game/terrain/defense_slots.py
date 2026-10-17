import json
from dataclasses import dataclass
from pathlib import Path

import pygame


@dataclass(slots=True)
class DefenseSlot:
    """A location where a defense element can be placed."""

    slot_id: int
    rect: pygame.Rect
    occupied: bool = False

    @property
    def center(self) -> pygame.Vector2:
        return pygame.Vector2(self.rect.center)


class DefenseSlots:
    """Loads and renders defense slots defined in a Tiled map."""

    def __init__(
        self,
        map_path: Path,
        slot_image_path: Path,
        *,
        load_image: bool = True,
    ) -> None:
        map_path = map_path.resolve()
        slot_image_path = slot_image_path.resolve()

        if not map_path.is_file():
            raise FileNotFoundError(f"Defense slot map not found: {map_path}")

        if not slot_image_path.is_file():
            raise FileNotFoundError(f"Defense slot image not found: {slot_image_path}")

        with map_path.open("r", encoding="utf-8") as file:
            map_data = json.load(file)

        self.map_width = int(map_data["width"] * map_data["tilewidth"])
        self.map_height = int(map_data["height"] * map_data["tileheight"])

        self.tile_width = int(map_data["tilewidth"])
        self.tile_height = int(map_data["tileheight"])

        self.slot_image = (
            pygame.image.load(str(slot_image_path)).convert_alpha() if load_image else None
        )

        self.slots: list[DefenseSlot] = []
        self._load_slots(map_data.get("layers", []))

    def _load_slots(self, layers: list[dict]) -> None:
        """Read rectangular slot objects from the Tiled slots layer."""
        slot_layer = next(
            (
                layer
                for layer in layers
                if layer.get("type") == "objectgroup"
                and layer.get("name", "").strip().lower() == "slots"
            ),
            None,
        )

        if slot_layer is None:
            raise ValueError(
                "No object layer named 'slots' was found in the Tiled map."
            )

        layer_x = float(slot_layer.get("x", 0)) * self.tile_width
        layer_y = float(slot_layer.get("y", 0)) * self.tile_height
        offset_x = float(slot_layer.get("offsetx", 0))
        offset_y = float(slot_layer.get("offsety", 0))

        for obj in slot_layer.get("objects", []):
            width = float(obj.get("width", 0))
            height = float(obj.get("height", 0))

            if width <= 0 or height <= 0:
                continue

            if obj.get("rotation", 0) != 0:
                raise ValueError(
                    f"Slot object {obj.get('id')} is rotated. "
                    "Only rectangular, unrotated slots are supported."
                )

            x = float(obj.get("x", 0)) + layer_x + offset_x
            y = float(obj.get("y", 0)) + layer_y + offset_y

            self.slots.append(
                DefenseSlot(
                    slot_id=int(obj["id"]),
                    rect=pygame.Rect(
                        round(x),
                        round(y),
                        round(width),
                        round(height),
                    ),
                )
            )

        self.slots.sort(key=lambda slot: slot.slot_id)

        if not self.slots:
            raise ValueError("No rectangular defense slots were found.")

    def draw(
        self,
        surface: pygame.Surface,
        show_ids: bool = False,
    ) -> None:
        """Draw slot icons, scaling them to the map and window."""
        if self.slot_image is None:
            raise RuntimeError("Slot images were not loaded for rendering.")
        scale_x = surface.get_width() / self.map_width
        scale_y = surface.get_height() / self.map_height

        overlay = pygame.Surface(
            surface.get_size(),
            pygame.SRCALPHA,
        )
        font = pygame.font.Font(None, 18) if show_ids else None

        for slot in self.slots:
            screen_rect = pygame.Rect(
                round(slot.rect.x * scale_x),
                round(slot.rect.y * scale_y),
                max(1, round(slot.rect.width * scale_x)),
                max(1, round(slot.rect.height * scale_y)),
            )

            if slot.occupied:
                # Occupied slots will display their defense element
                # instead of the empty-slot icon.
                continue

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
