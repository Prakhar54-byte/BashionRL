
import pygame

from src.game.config import load_config
from src.game.rendering.asset_loader import AssetLoader


def main() -> None:
    config = load_config()

    pygame.init()

    screen = pygame.display.set_mode(
        (config.window_width, config.window_height)
    )
    pygame.display.set_caption(config.window_title)

    assets = AssetLoader()
    grass = assets.load_texture(config.grass_path)
    road = assets.load_texture(config.road_path)

    grass = pygame.transform.scale(
        grass, (config.tile_size, config.tile_size)
    )
    road = pygame.transform.scale(
        road, (config.tile_size, config.tile_size)
    )

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((30, 32, 38))

        for row in range(config.rows):
            for col in range(config.cols):
                tile = grass

                # Temporary visual test: a road strip across the middle.
                if row == config.rows // 2:
                    tile = road

                screen.blit(
                    tile,
                    (col * config.tile_size, row * config.tile_size),
                )

        pygame.display.flip()
        clock.tick(config.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
