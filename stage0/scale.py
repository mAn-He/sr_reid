"""KPR 키포인트 스케일. PifPaf는 정수 좌표가 픽셀 중심인 규약을 쓴다(transforms/scale.py의 (w'-1)/(w-1), README 참조)."""


def scale_coord(v, k):
    return (v + 0.5) * k - 0.5


def scale_skeleton(skel, k):
    """(x, y, c)의 x, y만 바꾼다. 이미지 밖 좌표와 c=0 항목도 그대로 변환한다. bbox는 ltwh이다."""
    out = dict(skel)
    out["keypoints"] = [[scale_coord(x, k), scale_coord(y, k), c] for x, y, c in skel["keypoints"]]
    if "bbox" in skel:
        x, y, w, h = skel["bbox"]
        out["bbox"] = [scale_coord(x, k), scale_coord(y, k), w * k, h * k]
    return out
