"""Pygame UI drawing and world-click helpers for BastionRL."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pygame

from src.game.state.game_state import GameStatus

if TYPE_CHECKING:
    from src.game.simulation import Simulation


def draw_resource_slots(
    screen: pygame.Surface,
    simulation: Simulation,
) -> None:
    """Draw empty resource-slot icons using the shared slot texture."""
    manager = simulation.resource_slots
    slot_image = manager.slot_image

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
    """Draw a compact status bar and small corner panels.

    The old HUD used one tall, mostly opaque panel over the battlefield.
    This version packs stats into short chips at the top and moves controls
    to the bottom-left so most of the map remains visible.
    """
    state = simulation.state
    director = simulation.director
    screen_width, screen_height = screen.get_size()

    alive_enemies = sum(
        1 for enemy in simulation.enemies if enemy.alive and not enemy.reached_end
    )
    living_kings = [tower for tower in simulation.king_towers if tower.alive]
    king_hp = sum(max(0.0, tower.health) for tower in simulation.king_towers)
    king_max_hp = sum(tower.max_health for tower in simulation.king_towers)

    active_resource_towers = sum(
        1 for tower in simulation.resource_towers if tower.alive
    )
    resource_config = simulation.config.resource_tower
    tower_income = active_resource_towers * resource_config.production_amount
    passive_interval = simulation.config.passive_income_interval
    passive_income = simulation.config.passive_income_amount

    if selected_mode == "resource":
        selected_name = resource_config.name
        selected_cost = resource_config.cost
    else:
        tower_config = simulation.config.towers[selected_defense]
        selected_name = tower_config.name
        selected_cost = tower_config.cost

    # --- Top status bar -------------------------------------------------
    margin = 12
    panel_x = margin
    panel_y = margin
    panel_width = max(1, screen_width - margin * 2)
    chip_height = 26
    chip_gap = 7
    inner_x = panel_x + 10
    usable_width = panel_width - 20

    stats = [
        ("COINS", f"{state.resource_balance}", (238, 199, 92)),
        (
            "WAVE",
            f"{director.wave_number}/{simulation.balance.max_waves}",
            (130, 195, 255),
        ),
        ("ENEMIES", str(alive_enemies), (255, 150, 130)),
        ("SPAWNING", str(director.enemies_remaining_to_spawn), (238, 184, 112)),
        ("KILLS", str(state.enemies_killed), (145, 220, 155)),
        (
            "KINGS",
            f"{len(living_kings)}/{len(simulation.king_towers)}",
            (155, 205, 255),
        ),
        ("KING HP", f"{king_hp:.0f}/{king_max_hp:.0f}", (220, 220, 235)),
        ("MODE", difficulty.upper(), (205, 180, 255)),
    ]

    rendered_stats = []
    for label, value, accent in stats:
        text_surface = font.render(f"{label}  {value}", True, (242, 245, 250))
        chip_width = text_surface.get_width() + 20
        rendered_stats.append((text_surface, chip_width, accent))

    # Allow chips to wrap on smaller windows rather than run off-screen.
    rows: list[list[tuple[pygame.Surface, int, tuple[int, int, int]]]] = [[]]
    row_widths = [0]
    for item in rendered_stats:
        _, item_width, _ = item
        required = item_width if not rows[-1] else chip_gap + item_width
        if rows[-1] and row_widths[-1] + required > usable_width:
            rows.append([])
            row_widths.append(0)
            required = item_width
        rows[-1].append(item)
        row_widths[-1] += required

    summary = (
        f"RESOURCE TOWERS  {active_resource_towers}  ·  "
        f"+{tower_income}/{resource_config.production_interval:g}s"
        f"     PASSIVE  +{passive_income}/{passive_interval:g}s"
    )
    summary_surface = font.render(summary, True, (201, 214, 225))
    rows_height = len(rows) * (chip_height + 5)
    panel_height = 12 + rows_height + summary_surface.get_height() + 10

    panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel.fill((10, 15, 22, 190))
    screen.blit(panel, (panel_x, panel_y))
    pygame.draw.rect(
        screen,
        (87, 105, 125, 220),
        pygame.Rect(panel_x, panel_y, panel_width, panel_height),
        width=1,
        border_radius=10,
    )

    for row_index, row in enumerate(rows):
        chip_x = inner_x
        chip_y = panel_y + 8 + row_index * (chip_height + 5)
        for text_surface, chip_width, accent in row:
            chip_rect = pygame.Rect(chip_x, chip_y, chip_width, chip_height)
            pygame.draw.rect(screen, (31, 40, 51), chip_rect, border_radius=6)
            pygame.draw.rect(
                screen,
                accent,
                (chip_x, chip_y + 5, 3, chip_height - 10),
                border_radius=2,
            )
            screen.blit(
                text_surface,
                (chip_x + 11, chip_y + (chip_height - text_surface.get_height()) // 2),
            )
            chip_x += chip_width + chip_gap

    summary_y = panel_y + 8 + rows_height + 1
    # Keep this line legible even on small windows by clipping to the panel.
    available_summary_width = max(1, panel_width - 20)
    if summary_surface.get_width() > available_summary_width:
        summary_surface = pygame.transform.smoothscale(
            summary_surface,
            (available_summary_width, summary_surface.get_height()),
        )
    screen.blit(summary_surface, (inner_x, summary_y))

    # --- Compact build/control dock ------------------------------------
    if screen_width < 420:
        dock_width = max(1, screen_width - 24)
    else:
        dock_width = min(430, int(screen_width * 0.42))
    dock_height = 75 if not ai_mode else 56
    dock_x = 12
    dock_y = screen_height - dock_height - 12

    dock = pygame.Surface((dock_width, dock_height), pygame.SRCALPHA)
    dock.fill((10, 15, 22, 205))
    screen.blit(dock, (dock_x, dock_y))
    pygame.draw.rect(
        screen,
        (87, 105, 125),
        pygame.Rect(dock_x, dock_y, dock_width, dock_height),
        width=1,
        border_radius=10,
    )

    if ai_mode:
        title = font.render("AI SPECTATOR", True, (150, 205, 255))
        detail = font.render(
            "Watching the trained agent  ·  R restart  ·  ESC quit",
            True,
            (225, 230, 238),
        )
        screen.blit(title, (dock_x + 12, dock_y + 9))
        if detail.get_width() > dock_width - 24:
            detail = pygame.transform.smoothscale(
                detail, (max(1, dock_width - 24), detail.get_height())
            )
        screen.blit(detail, (dock_x + 12, dock_y + 32))
        return

    mode_label = (
        "RESOURCE TOWER" if selected_mode == "resource" else selected_name.upper()
    )
    title = font.render(
        f"BUILD: {mode_label}  ·  {selected_cost} COINS",
        True,
        (255, 220, 135),
    )
    controls = font.render(
        "Q Cannon   W Double Cannon   E Resource Tower",
        True,
        (226, 233, 241),
    )
    instructions = font.render(
        "Click slot: build  ·  Click destroyed King: repair  ·  R reset  ·  ESC quit",
        True,
        (165, 185, 205),
    )

    screen.blit(title, (dock_x + 12, dock_y + 7))
    available_text_width = dock_width - 24
    for text_surface, y in ((controls, dock_y + 29), (instructions, dock_y + 51)):
        if text_surface.get_width() > available_text_width:
            text_surface = pygame.transform.smoothscale(
                text_surface, (max(1, available_text_width), text_surface.get_height())
            )
        screen.blit(text_surface, (dock_x + 12, y))


def draw_wave_countdown_ui(
    screen: pygame.Surface,
    font: pygame.font.Font,
    title_font: pygame.font.Font,
    simulation: Simulation,
) -> None:
    """Show a small incoming-wave reminder in the bottom-right corner."""
    if simulation.state.is_terminal:
        return

    director = simulation.director
    if (
        director.wave_active
        or director.enemies_remaining_to_spawn > 0
        or simulation.enemies
    ):
        return

    next_wave = director.wave_number + 1
    if next_wave > simulation.balance.max_waves:
        return

    remaining = director.intermission_timer
    if remaining <= 0:
        return

    screen_width, screen_height = screen.get_size()
    panel_width = min(max(250, int(screen_width * 0.35)), 350, screen_width - 24)
    panel_height = 72
    panel = pygame.Rect(
        screen_width - panel_width - 12,
        screen_height - panel_height - 12,
        panel_width,
        panel_height,
    )

    panel_surface = pygame.Surface(panel.size, pygame.SRCALPHA)
    panel_surface.fill((17, 23, 32, 215))
    screen.blit(panel_surface, panel.topleft)
    pygame.draw.rect(
        screen,
        (218, 177, 80),
        panel,
        width=1,
        border_radius=10,
    )

    title_text = f"PREPARE FOR WAVE {next_wave}"
    title_surface = font.render(title_text, True, (255, 220, 125))
    if title_surface.get_width() > panel_width - 20:
        title_surface = pygame.transform.smoothscale(
            title_surface, (max(1, panel_width - 20), title_surface.get_height())
        )
    screen.blit(title_surface, (panel.x + 10, panel.y + 9))

    seconds_left = math.ceil(remaining)
    countdown_surface = title_font.render(
        f"Starts in {seconds_left}s", True, (245, 247, 250)
    )
    if countdown_surface.get_width() > panel_width - 20:
        countdown_surface = pygame.transform.smoothscale(
            countdown_surface,
            (max(1, panel_width - 20), countdown_surface.get_height()),
        )
    screen.blit(countdown_surface, (panel.x + 10, panel.y + 32))


def draw_terminal_overlay(
    screen: pygame.Surface,
    font: pygame.font.Font,
    large_font: pygame.font.Font,
    status: GameStatus,
    *,
    ai_finished: bool,
    king_defeat: bool,
) -> None:
    """Display a focused result card without fully obscuring the battlefield."""
    if status == GameStatus.WON:
        title_text = "VICTORY"
        title_color = (90, 230, 120)
        detail_text = "Your defenses held. The kingdom is safe."
    elif status == GameStatus.LOST and king_defeat:
        title_text = "KINGDOM FALLEN"
        title_color = (255, 100, 100)
        detail_text = "All King Towers have been destroyed."
    elif status == GameStatus.LOST:
        title_text = "BASE DESTROYED"
        title_color = (255, 100, 100)
        detail_text = "Your defenses have been overwhelmed."
    elif ai_finished:
        title_text = "TIME LIMIT"
        title_color = (255, 210, 90)
        detail_text = "The agent reached the configured decision limit."
    else:
        return

    screen_width, screen_height = screen.get_size()
    dim = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 105))
    screen.blit(dim, (0, 0))

    card_width = min(520, screen_width - 32)
    card_height = 190
    card = pygame.Rect(
        (screen_width - card_width) // 2,
        (screen_height - card_height) // 2,
        card_width,
        card_height,
    )
    card_surface = pygame.Surface(card.size, pygame.SRCALPHA)
    card_surface.fill((12, 18, 27, 238))
    screen.blit(card_surface, card.topleft)
    pygame.draw.rect(screen, title_color, card, width=2, border_radius=14)

    title = large_font.render(title_text, True, title_color)
    title_rect = title.get_rect(center=(card.centerx, card.y + 52))
    if title_rect.width > card.width - 24:
        title = pygame.transform.smoothscale(
            title, (card.width - 24, title.get_height())
        )
        title_rect = title.get_rect(center=(card.centerx, card.y + 52))
    screen.blit(title, title_rect)

    detail = font.render(detail_text, True, (218, 226, 236))
    if detail.get_width() > card.width - 28:
        detail = pygame.transform.smoothscale(
            detail, (card.width - 28, detail.get_height())
        )
    screen.blit(detail, detail.get_rect(center=(card.centerx, card.y + 103)))

    subtitle = font.render(
        "R  Restart episode     ·     ESC  Quit", True, (255, 255, 255)
    )
    screen.blit(subtitle, subtitle.get_rect(center=(card.centerx, card.y + 151)))


def _handle_world_click(
    simulation: Simulation,
    map_x: float,
    map_y: float,
    selected_defense: str,
    selected_mode: str,
) -> None:
    """Handle structure repairs and building placement in map coordinates."""
    click_position = pygame.Vector2(map_x, map_y)

    # King Towers have priority over placement slots.
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

        if (
            click_position.distance_squared_to(king_tower.position)
            <= hit_radius * hit_radius
        ):
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
