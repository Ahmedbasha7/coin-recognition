"""Стъпка 1: откриване на кръгли обекти (монети) в изображение.

Конвейер:
    1. зареждане на снимката
    2. преобразуване в сива скала
    3. размазване (за по-малко шум и фалшиви кръгове)
    4. Hough Circle Transform (cv2.HoughCircles)
    5. чертане на намерените кръгове върху копие на снимката

Пускане:
    python src/detect_circles.py images/moneti_2lv.jpg --min-radius 100 --max-radius 170 --min-dist 200
    python src/detect_circles.py images/moneti_present_all.jpg --min-radius 20 --max-radius 48 --min-dist 40

Радиусите и разстоянието зависят от размера на монетите в снимката, затова
се подават като параметри.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Не мога да заредя изображението: {path}")
    return image


def preprocess(image):
    """Сива скала + размазване."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.medianBlur(gray, 5)


def detect_circles(gray, min_radius, max_radius, min_dist, param1, param2):
    """Връща масив [[x, y, r], ...] или празен масив, ако няма кръгове."""
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return np.empty((0, 3), dtype=int)
    return np.round(circles[0]).astype(int)


def draw_circles(image, circles):
    result = image.copy()
    for i, (x, y, r) in enumerate(circles, start=1):
        cv2.circle(result, (x, y), r, (0, 255, 0), 2)    # контур
        cv2.circle(result, (x, y), 2, (0, 0, 255), 3)    # център
        cv2.putText(result, str(i), (x - 6, y - r - 4 if y - r > 12 else y + r + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)
    return result


def main():
    parser = argparse.ArgumentParser(description="Откриване на кръгли монети")
    parser.add_argument("image", help="път до входното изображение")
    parser.add_argument("--min-radius", type=int, default=20)
    parser.add_argument("--max-radius", type=int, default=0, help="0 = без ограничение")
    parser.add_argument("--min-dist", type=int, default=40, help="мин. разстояние между центровете")
    parser.add_argument("--param1", type=int, default=100, help="горен праг на Canny")
    parser.add_argument("--param2", type=int, default=50, help="праг на акумулатора (по-малко = повече кръгове)")
    parser.add_argument("--out", default="output", help="папка за резултата")
    args = parser.parse_args()

    image = load_image(args.image)
    gray = preprocess(image)
    circles = detect_circles(gray, args.min_radius, args.max_radius,
                             args.min_dist, args.param1, args.param2)

    print(f"Намерени кръгове: {len(circles)}")
    for i, (x, y, r) in enumerate(circles, start=1):
        print(f"  {i}: център=({x}, {y}), радиус={r}")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{Path(args.image).stem}_circles.png"
    cv2.imwrite(str(out_path), draw_circles(image, circles))
    print(f"Резултатът е записан в: {out_path}")


if __name__ == "__main__":
    main()
