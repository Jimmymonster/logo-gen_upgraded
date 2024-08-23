from PIL import Image, ImageDraw, ImageFilter
import random
import numpy as np

def gen_rainbow(width = 704, height = 576):

    image = Image.new('RGB', (width, height), 'black')
    draw = ImageDraw.Draw(image)

    def random_color():
        return tuple(random.randint(0, 255) for _ in range(3))

    for i in range(0, width, 10):
        color = random_color()
        draw.line([(i, 0), (i, height)], fill=color, width=10)

    image = image.filter(ImageFilter.GaussianBlur(radius=5))

    np_image = np.array(image)

    return np_image