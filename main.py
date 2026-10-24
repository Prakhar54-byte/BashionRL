
from __future__ import annotations

import argparse

import pygame

from src.game.config import load_config
from src.game.rendering.asset_loader import AssetLoader
from src.game.rendering.terrain_renderer import TerrainRenderer
from src.game.simulation import Simulation
from src.game.state.game_state import GameStatus


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


def draw_resource_slots(
    screen: pygame.Surface,
    simulation: Simulation,
) -> None:
    """Draw empty resource-slot icons using the shared slot texture."""
    manager = simulation.resource_slots
    slot_image = simulation.resource_slots.slot_image

    if slot_image is None:
        return

    scale_x = screen.get_width() / manager.map_width
    scale_y = screen.get_height() / manager.map_height

    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

    for slot in manager.slots:
        if slot.occupied:
            continue

        screen_rect = pygame.Rect(
            round(slot.rect.x * scale_x),
            round(slot.rect.y * scale_y),
            max(1, round(slot.rect.width * scale_x)),
            max(1, round(slot.rect.height * scale_y)),
        )

        icon = pygame.transform.smoothscale(
            slot_image,
            screen_rect.size,
        )
        overlay.blit(icon, screen_rect)

    screen.blit(overlay, (0, 0))


def draw_hud(
    screen: pygame.Surface,
    font: pygame.font.Font,
    simulation: Simulation,
    selected_defense: str,
    *,
    selected_mode: str,
    difficulty: str,
    ai_mode: bool,
) -> None:
    """Display episode status, economy, objectives, and building controls."""
    state = simulation.state
    director = simulation.director

    alive_enemies = sum(
        1
        for enemy in simulation.enemies
        if enemy.alive and not enemy.reached_end
    )

    living_kings = [
        tower
        for tower in simulation.king_towers
        if tower.alive
    ]

    king_health_text = " | ".join(
        f"{tower.health:.0f}"
        for tower in simulation.king_towers
    )

    active_resource_towers = sum(
        1 for tower in simulation.resource_towers if tower.alive
    )

    resource_config = simulation.config.resource_tower
    potential_income = (
        active_resource_towers * resource_config.production_amount
    )

    if selected_mode == "resource":
        selected_name = resource_config.name
        selected_cost = resource_config.cost
    else:
        tower_config = simulation.config.towers[selected_defense]
        selected_name = tower_config.name
        selected_cost = tower_config.cost

    health_fraction = max(
        0.0,
        min(1.0, state.base_health / state.max_base_health),
    )

    status_lines = [
        (
            f"Base HP (legacy): "
            f"{state.base_health:.0f}/{state.max_base_health:.0f}"
        ),
        f"Resources: {state.resource_balance}",
        (
            f"Wave: {director.wave_number}/{simulation.balance.max_waves}"
            f" | Difficulty: {difficulty}"
        ),
        f"Enemies alive: {alive_enemies}",
        f"Waiting to spawn: {director.enemies_remaining_to_spawn}",
        f"Total spawned: {state.enemies_spawned}",
        f"Kills: {state.enemies_killed} | Leaks: {state.enemies_leaked}",
        (
            f"King Towers alive: "
            f"{len(living_kings)}/{len(simulation.king_towers)}"
        ),
        f"King Tower HP: {king_health_text or 'None'}",
        (
            f"Resource Towers: {active_resource_towers}"
            f" | Potential income: +{potential_income}"
            f"/{resource_config.production_interval:g}s"
        ),
        f"Build: {selected_name} ({selected_cost} resources)",
        (
            "AI spectator"
            if ai_mode
            else "Q/W: Defense | E: Resource Tower"
        ),
        (
            "Click slot: Build | Click King: Repair"
            if not ai_mode
            else "Observing trained agent"
        ),
        "R: Reset | ESC: Quit",
    ]

    hud_width = min(610, screen.get_width() - 16)
    line_height = 23
    top_padding = 10
    health_bar_extra = 14
    hud_height = (
        top_padding
        + len(status_lines) * line_height
        + health_bar_extra
        + 8
    )

    hud_surface = pygame.Surface(
        (hud_width, hud_height),
        pygame.SRCALPHA,
    )
    hud_surface.fill((15, 18, 22, 215))
    screen.blit(hud_surface, (8, 8))

    # Legacy base-health bar is retained for compatibility with the current
    # Simulation leak mechanic.
    bar_x = 18
    bar_y = 42
    bar_width = 190
    bar_height = 10

    pygame.draw.rect(
        screen,
        (65, 35, 35),
        (bar_x, bar_y, bar_width, bar_height),
    )

    health_color = (
        (65, 210, 90)
        if health_fraction > 0.3
        else (225, 65, 65)
    )

    pygame.draw.rect(
        screen,
        health_color,
        (
            bar_x,
            bar_y,
            round(bar_width * health_fraction),
            bar_height,
        ),
    )

    for index, line in enumerate(status_lines):
        y_position = 15 + index * line_height

        if index > 0:
            y_position += health_bar_extra

        text_surface = font.render(
            line,
            True,
            (255, 255, 255),
        )

        screen.blit(text_surface, (18, y_position))


def draw_terminal_overlay(
    screen: pygame.Surface,
    font: pygame.font.Font,
    large_font: pygame.font.Font,
    status: GameStatus,
    *,
    ai_finished: bool,
    king_defeat: bool,
) -> None:
    """Display the episode result over the game."""
    if status == GameStatus.WON:
        title_text = "VICTORY"
        title_color = (90, 230, 120)
    elif status == GameStatus.LOST and king_defeat:
        title_text = "KINGDOM FALLEN"
        title_color = (255, 80, 80)
    elif status == GameStatus.LOST:
        title_text = "BASE DESTROYED"
        title_color = (255, 80, 80)
    elif ai_finished:
        title_text = "TIME LIMIT"
        title_color = (255, 210, 90)
    else:
        return

    overlay = pygame.Surface(
        screen.get_size(),
        pygame.SRCALPHA,
    )
    overlay.fill((0, 0, 0, 175))
    screen.blit(overlay, (0, 0))

    title = large_font.render(
        title_text,
        True,
        title_color,
    )

    subtitle = font.render(
        "Press R to restart or ESC to quit",
        True,
        (255, 255, 255),
    )

    center_x = screen.get_width() // 2
    center_y = screen.get_height() // 2

    screen.blit(
        title,
        title.get_rect(
            center=(center_x, center_y - 25),
        ),
    )

    screen.blit(
        subtitle,
        subtitle.get_rect(
            center=(center_x, center_y + 25),
        ),
    )


def _handle_world_click(
    simulation: Simulation,
    map_x: float,
    map_y: float,
    selected_defense: str,
    selected_mode: str,
) -> None:
    """Handle structure repairs and building placement in map coordinates."""
    click_position = pygame.Vector2(map_x, map_y)

    # King Towers have priority over placement slots. Clicking one requests
    # a repair through Simulation, which owns affordability and eligibility.
    for king_tower in simulation.king_towers:
        slot = next(
            (
                item
                for item in simulation.king_tower_slots.slots
                if item.slot_id == king_tower.structure_id
            ),
            None,
        )

        if slot is not None and slot.rect.collidepoint(
            round(map_x),
            round(map_y),
        ):
            simulation.start_king_tower_repair(king_tower.structure_id)
            return

        hit_radius = 40.0

        if click_position.distance_squared_to(
            king_tower.position
        ) <= hit_radius * hit_radius:
            simulation.start_king_tower_repair(king_tower.structure_id)
            return

    if selected_mode == "resource":
        for slot_index, slot in enumerate(simulation.resource_slots.slots):
            if slot.rect.collidepoint(round(map_x), round(map_y)):
                simulation.place_resource_tower(slot_index)
                return
    else:
        for slot_index, slot in enumerate(simulation.defense_slots.slots):
            if slot.rect.collidepoint(round(map_x), round(map_y)):
                simulation.place_tower(slot_index, selected_defense)
                return


def main() -> None:
    args = parse_args()
    config = load_config()

    pygame.init()

    try:
        screen = pygame.display.set_mode(
            (config.window_width, config.window_height)
        )
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

        # These lists are modified in place during reset, so the references
        # remain valid across manual and AI episode resets.
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

                    elif (
                        event.key == pygame.K_q
                        and "cannon" in defense_types
                    ):
                        selected_mode = "defense"
                        selected_defense = "cannon"

                    elif (
                        event.key == pygame.K_w
                        and "double_cannon" in defense_types
                    ):
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
                    map_x = (
                        mouse_x
                        * pathway.map_width
                        / screen.get_width()
                    )
                    map_y = (
                        mouse_y
                        * pathway.map_height
                        / screen.get_height()
                    )

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

                while (
                    ai_elapsed + 1e-12 >= simulation.timestep
                    and not ai_finished
                ):
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

            for enemy in enemies:
                enemy.draw(screen)

            draw_hud(
                screen,
                font,
                simulation,
                selected_defense,
                selected_mode=selected_mode,
                difficulty=args.difficulty,
                ai_mode=model is not None,
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
