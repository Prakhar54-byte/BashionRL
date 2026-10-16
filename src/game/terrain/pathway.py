import json
from pathlib import Path

import pygame


class Pathway:
    """A route defined by a named Tiled polyline object."""

    def __init__(self, map_path: Path) -> None:
        map_path = map_path.resolve()

        if not map_path.is_file():
            raise FileNotFoundError(f"Pathway map not found: {map_path}")

        with map_path.open("r", encoding="utf-8") as file:
            map_data = json.load(file)

        self.map_width = int(map_data["width"] * map_data["tilewidth"])
        self.map_height = int(map_data["height"] * map_data["tileheight"])

        self.points: list[pygame.Vector2] = []

        for layer in map_data.get("layers", []):
            if layer.get("type") != "objectgroup":
                continue

            for obj in layer.get("objects", []):
                if obj.get("name", "").strip().lower() != "pathway":
                    continue

                polyline = obj.get("polyline")

                if not polyline or len(polyline) < 2:
                    raise ValueError(
                        "The Tiled object 'pathway' must contain "
                        "at least two polyline points."
                    )

                object_x = float(obj.get("x", 0))
                object_y = float(obj.get("y", 0))

                self.points = [
                    pygame.Vector2(
                        object_x + float(point["x"]),
                        object_y + float(point["y"]),
                    )
                    for point in polyline
                ]

                break

            if self.points:
                break

        if len(self.points) < 2:
            raise ValueError(
                f"No polyline object named 'pathway' was found in {map_path}."
            )

    def draw(
        self,
        surface: pygame.Surface,
        color: tuple[int, int, int] = (255, 210, 60),
        width: int = 4,
        show_waypoints: bool = False,
    ) -> None:
        """Draw the route, scaled from map coordinates to the surface."""
        scale_x = surface.get_width() / self.map_width
        scale_y = surface.get_height() / self.map_height

        screen_points = [
            (
                round(point.x * scale_x),
                round(point.y * scale_y),
            )
            for point in self.points
        ]

        pygame.draw.lines(
            surface,
            color,
            False,
            screen_points,
            width,
        )

        if show_waypoints:
            for point in screen_points:
                pygame.draw.circle(
                    surface,
                    (255, 255, 255),
                    point,
                    5,
                )
