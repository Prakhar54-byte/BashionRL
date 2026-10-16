from __future__ import annotations

import pygame

from src.game.config import load_config
from src.game.entities.defense_tower import DefenseTower
from src.game.entities.enemy import Enemy, EnemyConfig
from src.game.rendering.asset_loader import AssetLoader
from src.game.rendering.terrain_renderer import TerrainRenderer
from src.game.state.game_state import GameState, GameStatus
from src.game.systems.wave_director import WaveConfig, WaveDirector
from src.game.terrain.defense_slots import DefenseSlots
from src.game.terrain.pathway import Pathway


def main() -> None:
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

        pathway = Pathway(config.pathway_path)

        defense_slots = DefenseSlots(
            map_path=config.pathway_path,
            slot_image_path=config.defense_slot_path,
        )

        enemy_sprites = {
            name: asset_loader.load_texture(path)
            for name, path in config.enemy_assets.items()
        }

        defense_sprites = {
            name: asset_loader.load_texture(path)
            for name, path in config.defense_assets.items()
        }

        if not enemy_sprites:
            raise ValueError("No enemy sprites are configured.")

        if not defense_sprites:
            raise ValueError("No defense sprites are configured.")

        # Enemy types are ordered from weakest to strongest.
        enemy_configs = {
            "enemy1": EnemyConfig(
                name="Walking Soldier",
                speed=90.0,
                max_health=100.0,
                size_scale=0.5,
                leak_damage=10.0,
                kill_reward=25,
            ),
            "enemy2": EnemyConfig(
                name="Advanced Soldier",
                speed=110.0,
                max_health=180.0,
                size_scale=0.5,
                leak_damage=10.0,
                kill_reward=40,
            ),
        }

        enemy_configs = {
            name: enemy_configs[name] for name in enemy_sprites if name in enemy_configs
        }

        if not enemy_configs:
            raise ValueError("No enemy configurations match the configured sprites.")

        tower_configs = {
            "cannon": {
                "name": "Cannon",
                "cost": 100,
                "range_radius": 180.0,
                "damage": 25.0,
                "fire_rate": 1.0,
                "projectile_speed": 350.0,
                "size_scale": 0.5,
            },
            "double_cannon": {
                "name": "Double Cannon",
                "cost": 150,
                "range_radius": 220.0,
                "damage": 20.0,
                "fire_rate": 2.0,
                "projectile_speed": 400.0,
                "size_scale": 0.5,
            },
        }

        defense_types = [name for name in tower_configs if name in defense_sprites]

        if not defense_types:
            raise ValueError(
                "No tower configurations match the configured defense sprites."
            )

        enemy_types = list(enemy_configs)
        selected_defense = defense_types[0]

        game_state = GameState(
            max_base_health=100.0,
            starting_resources=200,
        )

        enemies: list[Enemy] = []
        towers: list[DefenseTower] = []

        wave_config = WaveConfig(
            initial_enemy_count=5,
            additional_enemies_per_wave=2,
            spawn_interval=1.0,
            initial_intermission=3.0,
            intermission_between_waves=5.0,
            difficulty_growth=0.15,
        )

        wave_director = WaveDirector(
            enemy_types=enemy_types,
            config=wave_config,
            seed=42,
        )

        def spawn_enemy(enemy_type: str) -> None:
            """Spawn the enemy selected by the wave director."""
            if game_state.is_terminal:
                return

            if enemy_type not in enemy_configs:
                raise ValueError(f"Unknown enemy type requested: {enemy_type}")

            enemy = Enemy(
                pathway=pathway,
                sprite=enemy_sprites[enemy_type],
                config=enemy_configs[enemy_type],
            )

            enemies.append(enemy)
            game_state.register_spawn()

        def place_tower(slot) -> None:
            """Place a tower in an empty slot if resources allow it."""
            if game_state.is_terminal or slot.occupied:
                return

            tower_config = tower_configs[selected_defense]
            cost = tower_config["cost"]

            if not game_state.resources.can_afford(cost):
                return

            tower = DefenseTower(
                slot_id=slot.slot_id,
                position=slot.center,
                sprite=defense_sprites[selected_defense],
                range_radius=tower_config["range_radius"],
                damage=tower_config["damage"],
                fire_rate=tower_config["fire_rate"],
                projectile_speed=tower_config["projectile_speed"],
                size_scale=tower_config["size_scale"],
            )

            if not game_state.resources.spend(cost):
                return

            towers.append(tower)
            slot.occupied = True

        def reset_game() -> None:
            """Reset the episode and restart wave one."""
            nonlocal wave_director

            enemies.clear()
            towers.clear()
            game_state.reset()

            for slot in defense_slots.slots:
                slot.occupied = False

            wave_director = WaveDirector(
                enemy_types=enemy_types,
                config=wave_config,
                seed=42,
            )

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
                        reset_game()

                    elif event.key == pygame.K_q and "cannon" in defense_types:
                        selected_defense = "cannon"

                    elif event.key == pygame.K_w and "double_cannon" in defense_types:
                        selected_defense = "double_cannon"

                elif (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and not game_state.is_terminal
                ):
                    mouse_x, mouse_y = event.pos

                    map_x = mouse_x * pathway.map_width / screen.get_width()
                    map_y = mouse_y * pathway.map_height / screen.get_height()

                    for slot in defense_slots.slots:
                        if slot.rect.collidepoint(
                            round(map_x),
                            round(map_y),
                        ):
                            place_tower(slot)
                            break

            # Simulate the game only while the episode is active.
            if not game_state.is_terminal:
                # The wave director controls all enemy spawning.
                wave_director.update(
                    delta_time=delta_time,
                    active_enemy_count=sum(1 for enemy in enemies if enemy.alive),
                    spawn_enemy=spawn_enemy,
                )

                # Move living enemies along the pathway.
                for enemy in enemies:
                    if enemy.alive:
                        enemy.update(delta_time)

                # Process leaks before tower attacks. A leaked enemy
                # must not be counted as a kill or damage the base twice.
                for enemy in enemies:
                    if enemy.alive and enemy.reached_end:
                        game_state.register_leak(enemy.config.leak_damage)
                        enemy.alive = False

                # Towers attack living enemies.
                if not game_state.is_terminal:
                    for tower in towers:
                        tower.update(delta_time, enemies)

                # Count defeated enemies exactly once, after tower attacks.
                for enemy in enemies:
                    if not enemy.alive and not enemy.reached_end:
                        game_state.register_kill(enemy.config.kill_reward)

                # Remove defeated and leaked enemies.
                enemies[:] = [
                    enemy for enemy in enemies if enemy.alive and not enemy.reached_end
                ]

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
                f"Wave: {wave_director.wave_number}",
                (f"Enemies alive: {sum(1 for enemy in enemies if enemy.alive)}"),
                (f"Waiting to spawn: {wave_director.enemies_remaining_to_spawn}"),
                f"Total spawned: {game_state.enemies_spawned}",
                (
                    f"Kills: {game_state.enemies_killed} | "
                    f"Leaks: {game_state.enemies_leaked}"
                ),
                f"Selected tower: {selected_tower_config['name']}",
                f"Tower cost: {selected_tower_config['cost']}",
                "Q/W: Select tower | Click slot: Place",
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
            if game_state.status == GameStatus.LOST:
                overlay = pygame.Surface(
                    screen.get_size(),
                    pygame.SRCALPHA,
                )
                overlay.fill((0, 0, 0, 170))
                screen.blit(overlay, (0, 0))

                title = large_font.render(
                    "BASE DESTROYED",
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
