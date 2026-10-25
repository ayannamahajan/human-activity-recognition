from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
import os

font_dir = "/System/Library/Fonts/Supplemental/"
fonts_to_try = [
    "Brush Script.ttf",
    "Noteworthy.ttc",
    "SnellRoundhand.ttc",
    "Bradley Hand Bold.ttf",
    "Marker Felt.ttc",
    "Chalkduster.ttf",
    "Zapfino.ttf",
]

img = Image.new('RGB', (700, 900), color=(0, 0, 0))
draw = ImageDraw.Draw(img)

y = 40
for font_name in fonts_to_try:
    path = os.path.join(font_dir, font_name)
    if os.path.exists(path):
        try:
            font = ImageFont.truetype(path, 45)
            draw.text((20, y), f'Walking', font=font, fill=(0, 255, 255))
            draw.text((300, y + 5), font_name, fill=(150, 150, 150))
        except Exception as e:
            draw.text((20, y), f'{font_name} - error', fill=(255, 0, 0))
    else:
        draw.text((20, y), f'{font_name} - not found', fill=(255, 0, 0))
    y += 110

frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
cv2.imshow('Font Preview', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()