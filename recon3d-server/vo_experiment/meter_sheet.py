"""미터 크롭들을 프레임번호 라벨과 함께 격자 시트로 타일링(육안 판독용)."""
import glob

import cv2
import numpy as np

files = sorted(glob.glob("/tmp/vo_test/meter_*.png"))
COLS = 5
cell_w, cell_h = 380, 80
rows = (len(files) + COLS - 1) // COLS
sheet = np.zeros((rows * cell_h, COLS * cell_w, 3), np.uint8)
for i, fp in enumerate(files):
    im = cv2.imread(fp)
    fi = fp.split("_")[-1].split(".")[0]
    r, c = divmod(i, COLS)
    y, x = r * cell_h, c * cell_w
    h = min(im.shape[0], cell_h - 16)
    w = min(im.shape[1], cell_w - 100)
    sheet[y:y + h, x + 96:x + 96 + w] = im[:h, :w]
    cv2.putText(sheet, f"f{int(fi)}", (x + 4, y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
half = (rows + 1) // 2 * cell_h
cv2.imwrite("/tmp/vo_test/meter_sheet_1.png", sheet[:half])
cv2.imwrite("/tmp/vo_test/meter_sheet_2.png", sheet[half:])
print(f"{len(files)} crops -> 2 sheets ({rows} rows)")
