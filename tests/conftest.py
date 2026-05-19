import re
import zlib

import numpy as np
import pytest

from semantic_product_matcher import Listing


def bag_of_words_encoder(texts, dim=256):
    """Deterministic stand-in for a sentence-transformer: hashed word counts, L2-normalized.
    Texts sharing words get high cosine similarity — enough to test ranking offline."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"\w+", text.lower()):
            out[row, zlib.crc32(word.encode()) % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


@pytest.fixture
def ssd_listings():
    return [
        Listing(id="1", text="Samsung 990 PRO NVMe SSD 2TB"),
        Listing(id="2", text="Samsung 990 Pro 2000 GB M.2 NVMe drive"),
        Listing(id="3", text="Samsung 990 PRO NVMe SSD 1TB"),
        Listing(id="4", text="Logitech MX Master 3S wireless mouse"),
        Listing(id="5", text="Samsung 870 EVO 2TB SATA SSD"),
    ]
