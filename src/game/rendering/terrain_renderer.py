from pathlib import Path

import pygame

from game.rendering.asset_loader import AssetLoader


class TerrainRenderer:
    """Load and render the map stretched to fill the window."""

    def __init__(
        self,
        map_path: Path,
        asset_loader: AssetLoader,
    ) -> None:
        self.map_path = map_path.resolve()

        if not self.map_path.is_file():
            raise FileNotFoundError(f"Map image not found: {self.map_path}")

        self.image = asset_loader.load_texture(self.map_path)

        self.width = self.image.get_width()
        self.height = self.image.get_height()

        self.scaled_image: pygame.Surface | None = None
        self.window_size: tuple[int, int] | None = None

    def render(self, surface: pygame.Surface) -> None:
        """Stretch the map to cover the entire window."""

        window_size = surface.get_size()

        if window_size != self.window_size:
            self.scaled_image = pygame.transform.scale(
                self.image,
                window_size,
            )
            self.window_size = window_size

        surface.blit(self.scaled_image, (0, 0))
