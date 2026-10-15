import pygame

from src.game.config import GameConfig
from src.game.rendering.asset_loader import AssetLoader
from src.game.terrian.terrian_map import TerrianMap
from src.game.terrian.terrian_types import GRASS, ROAD


class TerrianRenderer:
    """
    Renders a Terrian Map using cached textures.
    """

    def __init__(
        self,
        config: GameConfig,
        asset_loader: AssetLoader,
    ) -> None:
        self.tile_size = config.tile_size

        self.textures = {
            GRASS: asset_loader.load_texture(config.grass_path),
            ROAD: asset_loader.load_texture(config.road_path),
        }

        self.textures = {
            terrian: pygame.transform.scale(
                texture, (self.tile_size, self.tile_size)
            )
            for terrian, texture in self.textures.items()
        }

    def render(self, surface: pygame.Surface, terrian_map: TerrianMap) -> None:
        for row in range(terrian_map.rows):
            for col in range(terrian_map.cols):
                cell = terrian_map.get_cell(row, col)
                texture = self.textures[cell.terrian]

                surface.blit(
                    texture,
                    (col * self.tile_size, row * self.tile_size)
                )

        
        