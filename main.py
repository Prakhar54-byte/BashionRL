from __future__ import annotations

import argparse
import pygame

from src.game.config import load_config
from src.game.entities.enemies import Tank
from src.game.rendering.asset_loader import AssetLoader
from src.game.rendering.terrain_renderer import TerrainRenderer
from src.game.rendering.ui import (
    _handle_world_click,
    draw_hud,
    draw_resource_slots,
    draw_terminal_overlay,
    draw_wave_countdown_ui,
)
from src.game.simulation import Simulation


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for manual and AI gameplay."""
    parser = argparse.ArgumentParser(
        description="Play BastionRL or watch a trained agent."
    )

    parser.add_argument(
        "--difficulty",
        choices=("baseline", "medium", "hard"),
        default="baseline",
        help="Difficulty configuration used for the episode.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=2400,
        help="Maximum number of RL decisions per episode.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when starting or resetting an episode.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a trained Maskable PPO checkpoint (.zip).",
    )

    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max-steps must be positive")

    return args


def main() -> None:
    args = parse_args()
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

            env = BastionEnv(
                args.difficulty,
                max_steps=args.max_steps,
                render_assets=True,
            )

            model = load_model(args.model, env)

            env.reset(seed=args.seed)
            simulation = env.simulation
        else:
            simulation = Simulation(
                args.difficulty,
                config=config,
                render_assets=True,
                seed=args.seed,
            )

        # These lists are modified in place during reset.
        pathway = simulation.pathway
        defense_slots = simulation.defense_slots
        game_state = simulation.state

        towers = simulation.towers
        resource_towers = simulation.resource_towers
        king_towers = simulation.king_towers
        enemies = simulation.enemies

        defense_types = simulation.tower_types

        if not defense_types:
            raise ValueError("No defense tower types are configured.")

        selected_defense = defense_types[0]
        selected_mode = "defense"

        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 23)
        large_font = pygame.font.Font(None, 64)
        wave_title_font = pygame.font.Font(None, 36)

        running = True

        ai_elapsed = 0.0
        ai_step = None
        ai_finished = False

        while running:
            delta_time = min(
                clock.tick(config.fps) / 1000.0,
                0.1,
            )

            # ----------------------------------------------------------
            # Input handling
            # ----------------------------------------------------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_r:
                        if env is not None:
                            env.reset(seed=args.seed)
                        else:
                            simulation.reset(seed=args.seed)

                        ai_elapsed = 0.0
                        ai_step = None
                        ai_finished = False

                    elif event.key == pygame.K_q and "cannon" in defense_types:
                        selected_mode = "defense"
                        selected_defense = "cannon"

                    elif event.key == pygame.K_w and "double_cannon" in defense_types:
                        selected_mode = "defense"
                        selected_defense = "double_cannon"

                    elif event.key == pygame.K_e:
                        selected_mode = "resource"

                elif (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and not game_state.is_terminal
                    and model is None
                ):
                    mouse_x, mouse_y = event.pos

                    # Convert window coordinates into map coordinates.
                    map_x = mouse_x * pathway.map_width / screen.get_width()
                    map_y = mouse_y * pathway.map_height / screen.get_height()

                    _handle_world_click(
                        simulation,
                        map_x,
                        map_y,
                        selected_defense,
                        selected_mode,
                    )

            # ----------------------------------------------------------
            # Simulation / trained agent
            # ----------------------------------------------------------
            if env is not None and model is not None and not ai_finished:
                ai_elapsed += delta_time

                while ai_elapsed + 1e-12 >= simulation.timestep and not ai_finished:
                    ai_elapsed -= simulation.timestep

                    if ai_step is None:
                        observation = env.observation()
                        action_mask = env.action_masks()

                        action, _ = model.predict(
                            observation,
                            deterministic=True,
                            action_masks=action_mask,
                        )

                        ai_step = env.step_ticks(int(action))

                    result = next(ai_step)

                    if result is not None:
                        (
                            _observation,
                            _reward,
                            terminated,
                            truncated,
                            _info,
                        ) = result

                        ai_finished = terminated or truncated
                        ai_step = None

            elif model is None:
                simulation.advance(delta_time)

            # ----------------------------------------------------------
            # Rendering
            # ----------------------------------------------------------
            terrain_renderer.render(screen)

            defense_slots.draw(screen, show_ids=False)
            draw_resource_slots(screen, simulation)

            for tower in towers:
                tower.draw(
                    screen,
                    map_width=pathway.map_width,
                    map_height=pathway.map_height,
                )

            for tower in resource_towers:
                tower.draw(
                    screen,
                    map_width=pathway.map_width,
                    map_height=pathway.map_height,
                )

            for tower in king_towers:
                tower.draw(
                    screen,
                    map_width=pathway.map_width,
                    map_height=pathway.map_height,
                )

            # Draw enemies and their projectiles.
            for enemy in enemies:
                enemy.draw(screen)

                if isinstance(enemy, Tank):
                    for projectile in enemy.projectiles:
                        projectile.draw(
                            screen,
                            map_width=pathway.map_width,
                            map_height=pathway.map_height,
                        )

            draw_hud(
                screen,
                font,
                simulation,
                selected_defense,
                selected_mode=selected_mode,
                difficulty=args.difficulty,
                ai_mode=model is not None,
            )

            # Automatic PvZ-style incoming-wave countdown.
            draw_wave_countdown_ui(
                screen,
                font,
                wave_title_font,
                simulation,
            )

            if game_state.is_terminal or ai_finished:
                draw_terminal_overlay(
                    screen,
                    font,
                    large_font,
                    game_state.status,
                    ai_finished=ai_finished,
                    king_defeat=game_state.all_king_towers_destroyed,
                )

            pygame.display.flip()

    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
