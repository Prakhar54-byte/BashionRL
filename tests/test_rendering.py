import math
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


def test_tower_cache_preserves_pixels_and_refreshes_on_resize(rendered_simulation, monkeypatch):
    sim = rendered_simulation
    sim.place_tower(0, "cannon")
    tower = sim.towers[0]
    monkeypatch.setattr(tower, "_draw_health_bar", lambda *args: None)
    original_scale = pygame.transform.smoothscale
    scale = Mock(wraps=original_scale)
    monkeypatch.setattr(pygame.transform, "smoothscale", scale)
    for size, angle in [((960, 640), 0), ((960, 640), 45), ((960, 640), 45), ((480, 320), 45)]:
        tower.angle = angle
        actual = pygame.Surface(size, pygame.SRCALPHA)
        expected = pygame.Surface(size, pygame.SRCALPHA)
        sx, sy = size[0] / sim.pathway.map_width, size[1] / sim.pathway.map_height
        base = original_scale(tower.base_sprite, (
            round(tower.base_sprite.get_width() * tower.base_size_scale * sx),
            round(tower.base_sprite.get_height() * tower.base_size_scale * sy),
        ))
        center = (round(tower.position.x * sx), round(tower.position.y * sy))
        expected.blit(base, base.get_rect(center=center))
        scaled = original_scale(tower.sprite, (round(tower.sprite.get_width() * tower.size_scale * sx),
                                               round(tower.sprite.get_height() * tower.size_scale * sy)))
        rotated = pygame.transform.rotate(scaled, angle)
        expected.blit(rotated, rotated.get_rect(center=(round(tower.position.x * sx),
                                                        round(tower.position.y * sy))))
        tower.draw(actual, sim.pathway.map_width, sim.pathway.map_height)
        assert pygame.image.tobytes(actual, "RGBA") == pygame.image.tobytes(expected, "RGBA")
    assert scale.call_count == 4


def test_enemy_cache_preserves_pixels_and_refreshes_on_turns(rendered_simulation, monkeypatch):
    sim = rendered_simulation
    sim._spawn_enemy("enemy1")
    enemy = sim.enemies[0]
    monkeypatch.setattr(enemy, "_draw_health_bar", lambda *args: None)
    original_rotate = pygame.transform.rotate
    rotate = Mock(wraps=original_rotate)
    monkeypatch.setattr(pygame.transform, "rotate", rotate)
    for size, direction in [((960, 640), (1, 0)), ((960, 640), (1, 0)),
                            ((960, 640), (0, 1)), ((480, 320), (0, 1))]:
        enemy.direction = pygame.Vector2(direction)
        actual = pygame.Surface(size, pygame.SRCALPHA)
        expected = pygame.Surface(size, pygame.SRCALPHA)
        sx, sy = size[0] / sim.pathway.map_width, size[1] / sim.pathway.map_height
        rotated = original_rotate(enemy.sprite, math.degrees(math.atan2(-direction[1], direction[0])))
        scaled = pygame.transform.scale(rotated, (round(rotated.get_width() * sx * enemy.size_scale),
                                                  round(rotated.get_height() * sy * enemy.size_scale)))
        expected.blit(scaled, scaled.get_rect(center=(round(enemy.position.x * sx),
                                                      round(enemy.position.y * sy))))
        enemy.draw(actual)
        assert pygame.image.tobytes(actual, "RGBA") == pygame.image.tobytes(expected, "RGBA")
    assert rotate.call_count == 3


@pytest.mark.parametrize("kind", ["defense", "resource", "king"])
def test_structure_health_and_destroyed_state_are_visible(rendered_simulation, kind):
    sim = rendered_simulation
    sim.place_tower(0, "cannon")
    structure = {"defense": sim.towers[0], "resource": sim.resource_towers[0],
                 "king": sim.king_towers[0]}[kind]
    structure.position = pygame.Vector2(480, 320)
    screen = pygame.Surface((960, 640), pygame.SRCALPHA)
    structure.take_damage(structure.max_health / 2)
    structure.draw(screen, 960, 640)
    damaged = pygame.image.tobytes(screen, "RGBA")
    pixels = pygame.surfarray.array3d(screen)
    health_color = {"defense": (65, 220, 90), "resource": (65, 210, 85),
                    "king": (65, 220, 90)}[kind]
    assert ((pixels == health_color).all(axis=2)).any()
    assert ((pixels == (55, 25, 25)).all(axis=2)).any()
    structure.take_damage(structure.max_health)
    screen.fill((0, 0, 0, 0))
    structure.draw(screen, 960, 640)
    assert pygame.image.tobytes(screen, "RGBA") != damaged
    assert screen.get_at((480, 320)).a > 0
