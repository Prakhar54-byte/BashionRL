import math
from copy import copy
from unittest.mock import Mock

import pygame
import pytest

from src.game.simulation import Simulation


@pytest.fixture
def rendered_simulation(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    pygame.display.set_mode((960, 640))
    yield Simulation(render_assets=True, seed=42)
    pygame.quit()


def test_tower_cache_preserves_pixels_and_refreshes_on_resize(
    rendered_simulation, monkeypatch
):
    sim = rendered_simulation
    sim.place_tower(0, "cannon")
    tower = sim.towers[0]
    original_scale = pygame.transform.smoothscale
    scale = Mock(wraps=original_scale)
    monkeypatch.setattr(pygame.transform, "smoothscale", scale)

    for size, angle in [
        ((960, 640), 0),
        ((960, 640), 45),
        ((960, 640), 45),
        ((480, 320), 45),
    ]:
        tower.angle = angle
        actual = pygame.Surface(size, pygame.SRCALPHA)
        expected = pygame.Surface(size, pygame.SRCALPHA)
        tower.draw(actual, sim.pathway.map_width, sim.pathway.map_height)

        # Render an uncached copy as the reference. This includes the stationary
        # base and health bar added to the current tower renderer.
        reference = copy(tower)
        reference._scaled_sprite = None
        reference._scaled_sprite_size = None
        reference._rotated_sprite = None
        reference._draw_angle = None
        reference._scaled_base_sprite = None
        reference._scaled_base_size = None

        monkeypatch.setattr(pygame.transform, "smoothscale", original_scale)
        reference.draw(expected, sim.pathway.map_width, sim.pathway.map_height)
        monkeypatch.setattr(pygame.transform, "smoothscale", scale)

        assert pygame.image.tobytes(actual, "RGBA") == pygame.image.tobytes(
            expected, "RGBA"
        )

    # Initial draw and resize each scale the base and barrel once.
    assert scale.call_count == 4

def test_enemy_cache_preserves_pixels_and_refreshes_on_turns(
    rendered_simulation, monkeypatch
):
    sim = rendered_simulation
    sim._spawn_enemy("enemy1")
    enemy = sim.enemies[0]
    monkeypatch.setattr(enemy, "_draw_health_bar", lambda *args: None)
    original_rotate = pygame.transform.rotate
    rotate = Mock(wraps=original_rotate)
    monkeypatch.setattr(pygame.transform, "rotate", rotate)
    for size, direction in [
        ((960, 640), (1, 0)),
        ((960, 640), (1, 0)),
        ((960, 640), (0, 1)),
        ((480, 320), (0, 1)),
    ]:
        enemy.direction = pygame.Vector2(direction)
        actual = pygame.Surface(size, pygame.SRCALPHA)
        expected = pygame.Surface(size, pygame.SRCALPHA)
        sx, sy = size[0] / sim.pathway.map_width, size[1] / sim.pathway.map_height
        rotated = original_rotate(
            enemy.sprite, math.degrees(math.atan2(-direction[1], direction[0]))
        )
        scaled = pygame.transform.scale(
            rotated,
            (
                round(rotated.get_width() * sx * enemy.size_scale),
                round(rotated.get_height() * sy * enemy.size_scale),
            ),
        )
        expected.blit(
            scaled,
            scaled.get_rect(
                center=(round(enemy.position.x * sx), round(enemy.position.y * sy))
            ),
        )
        enemy.draw(actual)
        assert pygame.image.tobytes(actual, "RGBA") == pygame.image.tobytes(
            expected, "RGBA"
        )
    assert rotate.call_count == 3
