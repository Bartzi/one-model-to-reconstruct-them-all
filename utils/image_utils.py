from PIL import Image, ImageDraw


def render_text_on_image(text: str, image: Image) -> Image:
    draw = ImageDraw.Draw(image)

    font = draw.getfont()
    text_size = draw.textsize(text, font=font)
    text_location = (image.width - text_size[0], image.height - text_size[1], image.width, image.height)
    draw.rectangle(text_location, fill=(255, 255, 255, 128))
    draw.text(text_location[:2], text, font=font, fill=(0, 255, 0))

    return image
