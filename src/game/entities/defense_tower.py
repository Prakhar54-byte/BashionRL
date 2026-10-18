import math

import pygame

from src.game.entities.enemy import Enemy


class Projectile:
    def __init__(
        self,
        position: pygame.Vector2,
        target: Enemy,
        speed: float,
        damage: float,
    ) -> None:
        self.position = position.copy()
        self.target = target
        self.speed = speed
        self.damage = damage
        self.alive = True

    def update(self, delta_time: float) -> None:
        if not self.alive:
            return

        if not self.target.alive or self.target.reached_end:
            self.alive = False
            return

        offset = self.target.position - self.position
        distance = offset.length()
        travel_distance = self.speed * max(0.0, delta_time)

        if distance <= max(6.0, travel_distance):
            self.target.take_damage(self.damage)
            self.alive = False
            return

        if distance > 0:
            self.position += offset / distance * travel_distance

    def draw(
        self,
        surface: pygame.Surface,
        map_width: float,
        map_height: float,
    ) -> None:
        if not self.alive or map_width <= 0 or map_height <= 0:
            return

        scale_x = surface.get_width() / map_width
        scale_y = surface.get_height() / map_height

        point = (
            round(self.position.x * scale_x),
            round(self.position.y * scale_y),
        )

        pygame.draw.circle(surface, (255, 220, 90), point, 4)
        pygame.draw.circle(surface, (255, 255, 220), point, 2)


class DefenseTower:
    def __init__(
        self,
        slot_id: int,
        position: pygame.Vector2,
        sprite: pygame.Surface,
        range_radius: float = 180.0,
        damage: float = 25.0,
        fire_rate: float = 1.0,
        projectile_speed: float = 350.0,
        size_scale: float = 3.0,
        slot_size: tuple[float, float] = (64.0, 64.0),
    ) -> None:
        self.slot_id = slot_id
        self.position = pygame.Vector2(position)
        self.sprite = sprite

        self.range_radius = range_radius
        self.damage = damage
        self.fire_rate = fire_rate
        self.projectile_speed = projectile_speed

        # Multiplier applied directly to the source sprite's dimensions.
        self.size_scale = max(0.01, size_scale)
        self.slot_size = (
            float(slot_size[0]),
            float(slot_size[1]),
        )

        self.cooldown = 0.0
        self.angle = 0.0
        self.projectiles: list[Projectile] = []
        self._scaled_sprite = None
        self._rotated_sprite = None
        self._draw_angle = None

    def _find_target(
        self,
        enemies: list[Enemy],
    ) -> Enemy | None:
        target = None
        nearest_distance = self.range_radius

        for enemy in enemies:
            if not enemy.alive or enemy.reached_end:
                continue

            distance = self.position.distance_to(enemy.position)

            if distance <= nearest_distance:
                nearest_distance = distance
                target = enemy

        return target

    def update(
        self,
        delta_time: float,
        enemies: list[Enemy],
    ) -> None:
        self.cooldown = max(0.0, self.cooldown - delta_time)

        target = self._find_target(enemies)

        if target is not None:
            direction = target.position - self.position

            if direction.length_squared() > 0:
                self.angle = math.degrees(math.atan2(-direction.y, direction.x)) - 90.0

            if self.cooldown <= 0.0:
                self.projectiles.append(
                    Projectile(
                        position=self.position,
                        target=target,
                        speed=self.projectile_speed,
                        damage=self.damage,
                    )
                )

                self.cooldown = 1.0 / self.fire_rate if self.fire_rate > 0 else 0.0

        for projectile in self.projectiles:
            projectile.update(delta_time)

        self.projectiles = [
            projectile for projectile in self.projectiles if projectile.alive
        ]

    def draw(
        self,
        surface: pygame.Surface,
        map_width: float,
        map_height: float,
        show_range: bool = False,
    ) -> None:
        if map_width <= 0 or map_height <= 0:
            return

        scale_x = surface.get_width() / map_width
        scale_y = surface.get_height() / map_height

        center = (
            round(self.position.x * scale_x),
            round(self.position.y * scale_y),
        )

        if show_range:
            range_surface = pygame.Surface(
                surface.get_size(),
                pygame.SRCALPHA,
            )

            range_width = round(self.range_radius * scale_x)
            range_height = round(self.range_radius * scale_y)

            pygame.draw.ellipse(
                range_surface,
                (80, 180, 255, 35),
                (
                    center[0] - range_width,
                    center[1] - range_height,
                    range_width * 2,
                    range_height * 2,
                ),
            )

            pygame.draw.ellipse(
                range_surface,
                (80, 180, 255, 110),
                (
                    center[0] - range_width,
                    center[1] - range_height,
                    range_width * 2,
                    range_height * 2,
                ),
                width=1,
            )

            surface.blit(range_surface, (0, 0))

        source_width, source_height = self.sprite.get_size()

        if source_width <= 0 or source_height <= 0:
            return

        # Scale the original sprite directly, in map-space units.
        target_width = max(
            1,
            round(source_width * self.size_scale * scale_x),
        )
        target_height = max(
            1,
            round(source_height * self.size_scale * scale_y),
        )

        target_size = (target_width, target_height)
        if self._scaled_sprite is None or self._scaled_sprite.get_size() != target_size:
            self._scaled_sprite = pygame.transform.smoothscale(self.sprite, target_size)
            self._draw_angle = None
        if self._draw_angle != self.angle:
            self._rotated_sprite = pygame.transform.rotate(self._scaled_sprite, self.angle)
            self._draw_angle = self.angle
        rotated_sprite = self._rotated_sprite

        sprite_rect = rotated_sprite.get_rect(center=center)
        surface.blit(rotated_sprite, sprite_rect)

        for projectile in self.projectiles:
            projectile.draw(surface, map_width, map_height)
