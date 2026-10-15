
from pathlib import Path

import pygame


class AssetLoader:
    """Loads and caches image assets."""

    def __init__(self) -> None:
        self._textures: dict[Path, pygame.Surface] = {}

    def load_texture(self, path: Path) -> pygame.Surface:
        path = path.resolve()

        if path in self._textures:
            return self._textures[path]

        if not path.is_file():
            raise FileNotFoundError(f"Texture not found: {path}")

        try:
            texture = pygame.image.load(str(path)).convert_alpha()
        except pygame.error as exc:
            raise RuntimeError(f"Failed to load texture: {path}") from exc

        self._textures[path] = texture
        return texture
