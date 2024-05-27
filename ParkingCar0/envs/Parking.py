import pygame


class Parking:
    def __init__(self, x, y, fill_color=pygame.Color("grey"), border_color=pygame.Color("grey")):
        self.x = x
        self.y = y
        self.border_thickness = 3
        self.fill_color = fill_color
        self.border_color = border_color

        self.slots = 6
        self.slot_width = 20
        self.slot_height = 40

    def draw(self, surface):
        up_line_width = (self.border_thickness + self.slot_width) * self.slots + self.border_thickness
        pygame.draw.line(surface, self.border_color, (self.x, self.y), (self.x + up_line_width, self.y),
                         self.border_thickness)  # Górna linia

        x = self.x + self.border_thickness
        y = self.y - self.border_thickness
        for slot in range(self.slots):
            pygame.draw.rect(surface, self.fill_color, (x, y, self.slot_width, self.slot_height))
            x += self.slot_width
            pygame.draw.line(surface, self.border_color, (x, y), (x, y + self.slot_height))
            x += self.border_thickness
