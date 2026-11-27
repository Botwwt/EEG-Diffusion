from PIL import Image
import pytest

from paper_spatial_photo_targets import crop_boxes, registered_oracle_members


def test_overlapping_crops_keep_raster_order_and_shape():
    image = Image.new("RGB", (500, 500))
    boxes = crop_boxes(image.size)
    assert boxes == [(0, 0, 350, 350), (150, 0, 500, 350),
                     (0, 150, 350, 500), (150, 150, 500, 500)]
    assert [image.crop(box).size for box in boxes] == [(350, 350)] * 4
    with pytest.raises(ValueError, match="square"):
        crop_boxes((500, 400))


def test_oracle_photos_require_same_order_as_eeg():
    archive = ["test_images/00001_cat/cat_1.jpg", "test_images/00002_dog/dog_1.jpg"]
    data = {"ids": ["cat_1", "dog_1"], "paths": archive}
    assert registered_oracle_members(data, {"ids": data["ids"]}, [1, 0]) == archive[::-1]
    with pytest.raises(ValueError, match="identity"):
        registered_oracle_members(data, {"ids": data["ids"][::-1]}, [1, 0])
