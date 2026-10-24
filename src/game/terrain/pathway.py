
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pygame


class Pathway:
    """A single route defined by a Tiled polyline object.

    Supports the current map's route naming convention:
        soldier_path1, soldier_path2, ...
        tank_path1, tank_path2, ...

    The legacy object name 'pathway' is also supported.
    """

    def __init__(
        self,
        map_path: Path,
        route_name: str | None = None,
    ) -> None:
        map_path = Path(map_path).resolve()
        map_data = self._read_map(map_path)

        routes = self._find_routes(map_data)

        if not routes:
            raise ValueError(
                f"No supported pathway objects were found in {map_path}. "
                "Expected names such as 'pathway', 'soldier_path1', "
                "or 'tank_path1'."
            )

        if route_name is not None:
            selected = next(
                (
                    route
                    for route in routes
                    if route[0].casefold() == route_name.casefold()
                ),
                None,
            )

            if selected is None:
                available = ", ".join(route[0] for route in routes)
                raise ValueError(
                    f"Route '{route_name}' was not found in {map_path}. "
                    f"Available routes: {available}"
                )
        else:
            # Preserve the original single-path behavior where possible.
            selected = next(
                (
                    route
                    for route in routes
                    if route[0].casefold() == "pathway"
                ),
                None,
            )

            # With the new map, the first soldier route is the default.
            if selected is None:
                selected = next(
                    (
                        route
                        for route in routes
                        if route[0].casefold().startswith("soldier_path")
                    ),
                    routes[0],
                )

        self._initialize(map_path, map_data, selected)

    @staticmethod
    def _read_map(map_path: Path) -> dict[str, Any]:
        """Load and validate a Tiled JSON map."""
        if not map_path.is_file():
            raise FileNotFoundError(
                f"Pathway map not found: {map_path}"
            )

        with map_path.open("r", encoding="utf-8") as file:
            map_data = json.load(file)

        required_fields = (
            "width",
            "height",
            "tilewidth",
            "tileheight",
            "layers",
        )

        missing = [
            field
            for field in required_fields
            if field not in map_data
        ]

        if missing:
            raise ValueError(
                f"Tiled map is missing required fields: {missing}"
            )

        for field in ("width", "height", "tilewidth", "tileheight"):
            value = map_data[field]

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"Tiled map field '{field}' must be positive."
                )

        return map_data

    @staticmethod
    def _find_routes(
        map_data: dict[str, Any],
    ) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
        """Find route polylines regardless of the layer's visibility."""
        routes: list[
            tuple[str, dict[str, Any], dict[str, Any]]
        ] = []

        for layer in map_data.get("layers", []):
            if layer.get("type") != "objectgroup":
                continue

            for obj in layer.get("objects", []):
                name = str(obj.get("name", "")).strip()
                normalized_name = name.casefold()

                is_route = (
                    normalized_name == "pathway"
                    or normalized_name.startswith("pathway_")
                    or normalized_name.startswith("soldier_path")
                    or normalized_name.startswith("tank_path")
                )

                if not is_route:
                    continue

                polyline = obj.get("polyline")

                if not isinstance(polyline, list) or len(polyline) < 2:
                    raise ValueError(
                        f"Route '{name}' must contain at least "
                        "two polyline points."
                    )

                routes.append((name, obj, layer))

        names = [name.casefold() for name, _, _ in routes]

        if len(names) != len(set(names)):
            raise ValueError(
                "Route object names must be unique, ignoring case."
            )

        return routes

    def _initialize(
        self,
        map_path: Path,
        map_data: dict[str, Any],
        route: tuple[str, dict[str, Any], dict[str, Any]],
    ) -> None:
        """Convert a Tiled polyline into map-space coordinates."""
        route_name, obj, layer = route

        self.map_path = map_path
        self.route_name = route_name

        if route_name.casefold().startswith("soldier_path"):
            self.route_family = "soldier"
        elif route_name.casefold().startswith("tank_path"):
            self.route_family = "tank"
        else:
            self.route_family = "default"

        # Convenient identifier to assign to spawned enemies.
        self.lane_id = route_name

        self.map_width = int(
            map_data["width"] * map_data["tilewidth"]
        )
        self.map_height = int(
            map_data["height"] * map_data["tileheight"]
        )

        tile_width = float(map_data["tilewidth"])
        tile_height = float(map_data["tileheight"])

        # Tiled object-group x/y positions are expressed in tiles.
        # Pixel offsets are expressed in pixels.
        layer_x = float(layer.get("x", 0)) * tile_width
        layer_y = float(layer.get("y", 0)) * tile_height

        layer_x += float(layer.get("offsetx", 0))
        layer_y += float(layer.get("offsety", 0))

        object_x = float(obj.get("x", 0)) + layer_x
        object_y = float(obj.get("y", 0)) + layer_y

        rotation = float(obj.get("rotation", 0))

        if not math.isfinite(rotation):
            raise ValueError(
                f"Route '{route_name}' has an invalid rotation."
            )

        angle = math.radians(rotation)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        self.points: list[pygame.Vector2] = []

        for index, point in enumerate(obj["polyline"]):
            try:
                local_x = float(point["x"])
                local_y = float(point["y"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Route '{route_name}' has an invalid point "
                    f"at index {index}."
                ) from exc

            if not (
                math.isfinite(local_x)
                and math.isfinite(local_y)
            ):
                raise ValueError(
                    f"Route '{route_name}' has non-finite "
                    f"coordinates at point {index}."
                )

            # Apply rotation about the Tiled object's origin.
            rotated_x = (
                local_x * cos_angle - local_y * sin_angle
            )
            rotated_y = (
                local_x * sin_angle + local_y * cos_angle
            )

            world_point = pygame.Vector2(
                object_x + rotated_x,
                object_y + rotated_y,
            )

            if (
                world_point.x < -1e-6
                or world_point.y < -1e-6
                or world_point.x > self.map_width + 1e-6
                or world_point.y > self.map_height + 1e-6
            ):
                raise ValueError(
                    f"Route '{route_name}' point {index} lies "
                    f"outside map bounds: "
                    f"({world_point.x:.1f}, {world_point.y:.1f})."
                )

            self.points.append(world_point)

        for index in range(len(self.points) - 1):
            if self.points[index].distance_squared_to(
                self.points[index + 1]
            ) <= 1e-12:
                raise ValueError(
                    f"Route '{route_name}' has duplicate consecutive "
                    f"points at index {index}."
                )

    @classmethod
    def load_routes(
        cls,
        map_path: Path,
        *,
        route_family: str = "soldier",
    ) -> list[Pathway]:
        """Load every route belonging to a specific enemy family.

        Args:
            map_path: Tiled JSON map.
            route_family: 'soldier', 'tank', or 'all'.

        Returns:
            A list of independently usable Pathway objects.
        """
        map_path = Path(map_path).resolve()
        normalized_family = route_family.strip().casefold()

        valid_families = {"soldier", "tank", "all"}

        if normalized_family not in valid_families:
            raise ValueError(
                f"Unknown route family '{route_family}'. "
                f"Choose one of: {', '.join(sorted(valid_families))}."
            )

        map_data = cls._read_map(map_path)
        routes = cls._find_routes(map_data)

        if normalized_family == "all":
            selected_routes = routes
        elif normalized_family == "soldier":
            selected_routes = [
                route
                for route in routes
                if route[0].casefold().startswith("soldier_path")
            ]
        else:
            selected_routes = [
                route
                for route in routes
                if route[0].casefold().startswith("tank_path")
            ]

        if not selected_routes:
            raise ValueError(
                f"No routes were found for family '{normalized_family}' "
                f"in {map_path}."
            )

        pathways: list[Pathway] = []

        for route in selected_routes:
            pathway = cls.__new__(cls)
            pathway._initialize(map_path, map_data, route)
            pathways.append(pathway)

        return pathways

    def draw(
        self,
        surface: pygame.Surface,
        color: tuple[int, int, int] = (255, 210, 60),
        width: int = 4,
        show_waypoints: bool = False,
    ) -> None:
        """Draw the route after converting map coordinates to screen space."""
        if self.map_width <= 0 or self.map_height <= 0:
            return

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
