from PIL import Image

tags = ['U0.png', 'US1.png', 'UC1.png', 'US2.png', 'UC2.png', 'US3.png', 'UC3.png']
number = 0
for tag in tags:
    # Open your small image
    img = Image.open(f"{number}/{tag}")

    # Scale factor (e.g., 4x bigger)
    scale_factor = 16
    new_size = (img.width * scale_factor, img.height * scale_factor)

    # Resize using nearest neighbor
    resized_img = img.resize(new_size, resample=Image.NEAREST)

    # Save or display
    resized_img.save(f"{number}/resized/{tag}")
    resized_img.show()