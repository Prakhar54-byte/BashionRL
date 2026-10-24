from __future__ import annotations

import pygame
import pytest

from src.game.rendering.ui import (
    _handle_world_click,
    draw_hud,
    draw_resource_slots,
    draw_terminal_overlay,
    draw_wave_countdown_ui,
)
from src.game.simulation import Simulation
from src.game.state.game_state import GameStatus


@pytest.fixture
def ui_context(monkeypatch):
    """Create a rendered simulation and a headless Pygame surface."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    pygame.display.set_mode((1280, 720))

    sim = Simulation(seed=17, render_assets=True)
    screen = pygame.Surface((1280, 720), pygame.SRCALPHA)
    font = pygame.font.Font(None, 23)
    title_font = pygame.font.Font(None, 36)

    yield sim, screen, font, title_font

    pygame.quit()


def _pixels(surface: pygame.Surface) -> bytes:
    return pygame.image.tobytes(surface, "RGBA")


def test_draw_resource_slots_is_safe_for_empty_and_occupied_slots(ui_context):
    sim, screen, _, _ = ui_context

    draw_resource_slots(screen, sim)
    empty_slot = next(
        (slot for slot in sim.resource_slots.slots if not slot.occupied),
        None,
    )
    if empty_slot is not None:
        empty_slot.occupied = True
        draw_resource_slots(screen, sim)

    assert screen.get_size() == (1280, 720)


@pytest.mark.parametrize("selected_mode", ["defense", "resource"])
def test_draw_hud_renders_manual_and_resource_build_modes(ui_context, selected_mode):
    sim, screen, font, _ = ui_context
    before = _pixels(screen)

    draw_hud(
        screen,
        font,
        sim,
        "cannon",
        selected_mode=selected_mode,
        difficulty="baseline",
        ai_mode=False,
    )

    assert _pixels(screen) != before


def test_draw_hud_renders_ai_spectator_mode(ui_context):
    sim, screen, font, _ = ui_context
    before = _pixels(screen)

    draw_hud(
        screen,
        font,
        sim,
        "cannon",
        selected_mode="defense",
        difficulty="hard",
        ai_mode=True,
    )

    assert _pixels(screen) != before


def test_wave_countdown_draws_during_intermission_but_not_during_wave(ui_context):
    sim, screen, font, title_font = ui_context
    director = sim.director
    director.wave_number = 0
    director.wave_active = False
    director.enemies_remaining_to_spawn = 0
    director.intermission_timer = 5.0
    sim.enemies.clear()

    before = _pixels(screen)
    draw_wave_countdown_ui(screen, font, title_font, sim)
    assert _pixels(screen) != before

    screen.fill((0, 0, 0, 0))
    during_wave_before = _pixels(screen)
    director.wave_active = True
    draw_wave_countdown_ui(screen, font, title_font, sim)
    assert _pixels(screen) == during_wave_before


def test_terminal_overlay_draws_for_won_episode(ui_context):
    _, screen, font, title_font = ui_context
    before = _pixels(screen)

    draw_terminal_overlay(
        screen,
        font,
        title_font,
        GameStatus.WON,
        ai_finished=False,
        king_defeat=False,
    )

    assert _pixels(screen) != before


def test_clicking_empty_defense_slot_places_selected_tower(ui_context):
    sim, _, _, _ = ui_context
    slot_index = next(
        i for i, slot in enumerate(sim.defense_slots.slots) if not slot.occupied
    )
    slot = sim.defense_slots.slots[slot_index]

    _handle_world_click(
        sim,
        slot.rect.centerx,
        slot.rect.centery,
        selected_defense="cannon",
        selected_mode="defense",
    )

    assert len(sim.towers) == 1
    assert sim.towers[0].tower_type == "cannon"


def test_clicking_empty_resource_slot_places_resource_tower(ui_context):
    sim, _, _, _ = ui_context
    slot_index = next(
        i for i, slot in enumerate(sim.resource_slots.slots) if not slot.occupied
    )
    slot = sim.resource_slots.slots[slot_index]

    _handle_world_click(
        sim,
        slot.rect.centerx,
        slot.rect.centery,
        selected_defense="cannon",
        selected_mode="resource",
    )

    assert len(sim.resource_towers) == 1
