"""Pygame UI helpers for explore.py (hold frame + re-run / quit)."""

from __future__ import annotations

import select
import sys
from typing import Literal

Action = Literal["rerun", "quit"]


class Button:
    def __init__(
        self,
        rect,
        label: str,
        color: tuple[int, int, int],
        hover_color: tuple[int, int, int],
    ) -> None:
        self.rect = rect
        self.label = label
        self.color = color
        self.hover_color = hover_color
        self.hovered = False

    def draw(self, surface, font) -> None:
        import pygame

        color = self.hover_color if self.hovered else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, (30, 30, 30), self.rect, width=2, border_radius=6)
        text = font.render(self.label, True, (255, 255, 255))
        text_rect = text.get_rect(center=self.rect.center)
        surface.blit(text, text_rect)

    def update_hover(self, mouse_pos) -> None:
        self.hovered = self.rect.collidepoint(mouse_pos)

    def clicked(self, event) -> bool:
        import pygame

        return (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        )


def wait_after_episode(env) -> Action:
    print("Episode over — click Re-run, press R, or close the window to play again.")
    print("Click Quit, press Q/Esc, or press Enter here to exit.")

    try:
        import pygame
    except ImportError:
        line = input("Press Enter to exit, or type r to re-run: ").strip().lower()
        return "rerun" if line == "r" else "quit"

    font = pygame.font.SysFont(None, 28)
    clock = pygame.time.Clock()

    while True:
        env.render()
        surface = pygame.display.get_surface()
        if surface is None:
            clock.tick(30)
            continue

        width, height = surface.get_size()
        btn_w, btn_h, gap, margin = 120, 40, 20, 20
        row_width = btn_w * 2 + gap
        x0 = (width - row_width) // 2
        y0 = height - btn_h - margin

        buttons = [
            Button(
                pygame.Rect(x0, y0, btn_w, btn_h),
                "Re-run",
                (46, 125, 50),
                (56, 142, 60),
            ),
            Button(
                pygame.Rect(x0 + btn_w + gap, y0, btn_w, btn_h),
                "Quit",
                (97, 97, 97),
                (117, 117, 117),
            ),
        ]

        mouse_pos = pygame.mouse.get_pos()
        for button in buttons:
            button.update_hover(mouse_pos)
            button.draw(surface, font)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return "rerun"
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return "quit"
            if buttons[0].clicked(event):
                return "rerun"
            if buttons[1].clicked(event):
                return "quit"

        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            return "quit"

        clock.tick(30)
