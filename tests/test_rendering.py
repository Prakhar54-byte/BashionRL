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


@pytest.mark.parametrize("kind", ["defense", "resource", "king"])
def test_damaged_and_destroyed_structures_remain_visible(rendered_simulation, kind):
    sim = rendered_simulation
    if kind == "defense":
        assert sim.place_tower(0, "cannon")
        target = sim.towers[0]
    elif kind == "resource":
        assert sim.place_resource_tower(0)
        target = sim.resource_towers[0]
    else:
        target = sim.king_towers[0]
    target.position = pygame.Vector2(480, 320)
    screen = pygame.Surface((960, 640), pygame.SRCALPHA)
    target.take_damage(target.max_health / 2)
    target.draw(screen, 960, 640)
    damaged = pygame.image.tobytes(screen, "RGBA")
    pixels = pygame.surfarray.array3d(screen)
    health_color = (65, 210, 85) if kind == "resource" else (65, 220, 90)
    assert ((pixels == health_color).all(axis=2)).any()
    assert ((pixels == (55, 25, 25)).all(axis=2)).any()
    target.destroy()
    screen.fill((0, 0, 0, 0))
    target.draw(screen, 960, 640)
    assert pygame.image.tobytes(screen, "RGBA") != damaged
    assert screen.get_at((480, 320)).a > 0


def test_tank_gun_aims_independently_of_chassis(rendered_simulation):
    sim = rendered_simulation
    sim._spawn_enemy("light_tank")
    tank = sim.enemies[0]
    tank.position = pygame.Vector2(480, 320)
    king = sim.king_towers[0]
    king.position = tank.position + pygame.Vector2(0, -100)
    direction = tank.direction.copy()
    before = pygame.Surface((960, 640), pygame.SRCALPHA)
    tank.draw(before)
    tank.update_combat(0, [], king_towers=[king])
    after = pygame.Surface((960, 640), pygame.SRCALPHA)
    tank.draw(after)
    assert tank.direction == direction and tank.gun_angle == pytest.approx(90)
    assert pygame.image.tobytes(before, "RGBA") != pygame.image.tobytes(after, "RGBA")
