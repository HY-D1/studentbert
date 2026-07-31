#!/usr/bin/env python3
"""Regenerate poster/qr_code.png (links the poster to the repo).
Requires opencv-contrib (cv2). Usage: python3 make_qr.py [url]"""
import sys
import cv2
import numpy as np

url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/HY-D1/studentbert"
params = cv2.QRCodeEncoder_Params()
params.correction_level = cv2.QRCODE_ENCODER_CORRECT_LEVEL_Q
enc = cv2.QRCodeEncoder_create(params)
qr = enc.encode(url)
scale = 40
big = np.kron(qr, np.ones((scale, scale), dtype=np.uint8))
quiet = 4 * scale
canvas = np.full((big.shape[0] + 2 * quiet, big.shape[1] + 2 * quiet), 255, dtype=np.uint8)
canvas[quiet:-quiet, quiet:-quiet] = np.where(big > 0, 255, 0)
cv2.imwrite("qr_code.png", canvas)
val, _, _ = cv2.QRCodeDetector().detectAndDecode(canvas)
assert val == url, f"self-check failed: {val!r}"
print(f"wrote qr_code.png ({canvas.shape[0]}px), decodes to {val}")
