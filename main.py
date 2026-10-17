from __future__ import annotations

import argparse

import pygame

from src.game.config import load_config
from src.game.rendering.asset_loader import AssetLoader
from src.game.rendering.terrain_renderer import TerrainRenderer
from src.game.simulation import Simulation
from src.game.state.game_state import GameStatus


def main() -> None:
    parser = argparse.ArgumentParser(description="Play BastionRL or watch a trained agent.")
    parser.add_argument("--difficulty", choices=("medium", "hard"), default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", help="Maskable PPO checkpoint (.zip)")
    args = parser.parse_args()
    config = load_config()
    pygame.init()

    try:
        screen = pygame.display.set_mode((config.window_width, config.window_height))
        pygame.display.set_caption(config.window_title)

        asset_loader = AssetLoader()

        terrain_renderer = TerrainRenderer(
            map_path=config.map_path,
            asset_loader=asset_loader,
        )

        env = None
        model = None
        if args.model:
            from src.game.rl.checkpoint import load_model
            from src.game.rl.environment import BastionEnv

            env = BastionEnv(args.difficulty, render_assets=True)
            model = load_model(args.model, env)
            env.reset(seed=args.seed)
            simulation = env.simulation
        else:
            simulation = Simulation(args.difficulty, config=config,
                                    render_assets=True, seed=args.seed)
        pathway = simulation.pathway
        defense_slots = simulation.defense_slots
        game_state = simulation.state
        towers = simulation.towers
        enemies = simulation.enemies
        tower_configs = config.towers
        defense_types = simulation.tower_types
        selected_defense = defense_types[0]
        ai_elapsed = 0.5
        ai_finished = False

        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 25)
        large_font = pygame.font.Font(None, 64)
        running = True

        while running:
            delta_time = clock.tick(config.fps) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_r:
                        if env:
                            env.reset(seed=args.seed)
                        else:
                            simulation.reset(seed=args.seed)
                        ai_elapsed = 0.5
                        ai_finished = False

                    elif event.key == pygame.K_q and "cannon" in defense_types:
                        selected_defense = "cannon"

                    elif event.key == pygame.K_w and "double_cannon" in defense_types:
                        selected_defense = "double_cannon"

                elif (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and not game_state.is_terminal
                    and model is None
                ):
                    mouse_x, mouse_y = event.pos

                    map_x = mouse_x * pathway.map_width / screen.get_width()
                    map_y = mouse_y * pathway.map_height / screen.get_height()

                    for slot_index, slot in enumerate(defense_slots.slots):
                        if slot.rect.collidepoint(
                            round(map_x),
                            round(map_y),
                        ):
                            simulation.place_tower(slot_index, selected_defense)
                            break

            if env and model and not ai_finished:
                ai_elapsed += delta_time
                if ai_elapsed >= 0.5:
                    ai_elapsed -= 0.5
                    action, _ = model.predict(env.observation(), deterministic=True,
                                              action_masks=env.action_masks())
                    _, _, terminated, truncated, _ = env.step(int(action))
                    ai_finished = terminated or truncated
            elif model is None:
                simulation.advance(delta_time)
            wave_director = simulation.director

            # Render the scene.
            terrain_renderer.render(screen)
            defense_slots.draw(screen, show_ids=False)

            for tower in towers:
                tower.draw(
                    screen,
                    map_width=pathway.map_width,
                    map_height=pathway.map_height,
                )

            for enemy in enemies:
                enemy.draw(screen)

            health_fraction = game_state.base_health / game_state.max_base_health

            selected_tower_config = tower_configs[selected_defense]

            status_lines = [
                (
                    f"Base HP: {game_state.base_health:.0f}/"
                    f"{game_state.max_base_health:.0f}"
                ),
                f"Resources: {game_state.resource_balance}",
                f"Wave: {wave_director.wave_number}/{simulation.balance.max_waves} | {args.difficulty}",
                (f"Enemies alive: {sum(1 for enemy in enemies if enemy.alive)}"),
                (f"Waiting to spawn: {wave_director.enemies_remaining_to_spawn}"),
                f"Total spawned: {game_state.enemies_spawned}",
                (
                    f"Kills: {game_state.enemies_killed} | "
                    f"Leaks: {game_state.enemies_leaked}"
                ),
                f"Selected tower: {selected_tower_config.name}",
                f"Tower cost: {selected_tower_config.cost}",
                "AI spectator" if model else "Q/W: Select tower | Click slot: Place",
                "R: Reset | ESC: Quit",
            ]

            hud_width = 440
            hud_height = len(status_lines) * 25 + 12

            hud_surface = pygame.Surface(
                (hud_width, hud_height),
                pygame.SRCALPHA,
            )
            hud_surface.fill((15, 18, 22, 200))
            screen.blit(hud_surface, (8, 8))

            # Draw base health bar.
            bar_x = 18
            bar_y = 39
            bar_width = 180
            bar_height = 10

            pygame.draw.rect(
                screen,
                (65, 35, 35),
                (bar_x, bar_y, bar_width, bar_height),
            )

            pygame.draw.rect(
                screen,
                ((65, 210, 90) if health_fraction > 0.3 else (225, 65, 65)),
                (
                    bar_x,
                    bar_y,
                    round(bar_width * health_fraction),
                    bar_height,
                ),
            )

            for index, line in enumerate(status_lines):
                y_position = 15 + index * 25
                if index > 0:
                    y_position += 12

                text_surface = font.render(
                    line,
                    True,
                    (255, 255, 255),
                )
                screen.blit(text_surface, (18, y_position))

            # Episode termination overlay.
            if game_state.is_terminal or ai_finished:
                overlay = pygame.Surface(
                    screen.get_size(),
                    pygame.SRCALPHA,
                )
                overlay.fill((0, 0, 0, 170))
                screen.blit(overlay, (0, 0))

                title = large_font.render(
                    ("VICTORY" if game_state.status == GameStatus.WON else
                     "BASE DESTROYED" if game_state.status == GameStatus.LOST else "TIME LIMIT"),
                    True,
                    (255, 80, 80),
                )
                subtitle = font.render(
                    "Press R to restart or ESC to quit",
                    True,
                    (255, 255, 255),
                )

                screen.blit(
                    title,
                    title.get_rect(
                        center=(
                            screen.get_width() // 2,
                            screen.get_height() // 2 - 25,
                        )
                    ),
                )
                screen.blit(
                    subtitle,
                    subtitle.get_rect(
                        center=(
                            screen.get_width() // 2,
                            screen.get_height() // 2 + 25,
                        )
                    ),
                )

            pygame.display.flip()

    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
