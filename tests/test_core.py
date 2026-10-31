"""test 가드, Market 규칙, PK 샘플러, 부트스트랩 판정 단위 테스트. 작은 합성 배열만 쓴다."""
import numpy as np
import pytest

from common import guard
from eval.bootstrap import resample_means, verdict
from eval.metrics_reid import evaluate_query
from reid.trainer import lr_at, pk_batches


@pytest.mark.parametrize("path", [
    f"{guard.LOCKED_ROOT}/query/a.jpg",
    f"{guard.MARKET_ROOT}/bounding_box_test/a.jpg",
    f"{guard.MARKET_ROOT}/query",
    f"{guard.MARKET_ROOT}/gt_bbox/x.jpg",
])
def test_guard_blocks_test_paths(path, monkeypatch, tmp_path):
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    monkeypatch.setattr(guard, "FLAG", str(tmp_path / "flag"))
    with pytest.raises(guard.GuardViolation):
        guard.check(path)
    assert (tmp_path / "flag").exists()


def test_guard_allows_train_and_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    monkeypatch.setattr(guard, "FLAG", str(tmp_path / "flag"))
    guard.check(f"{guard.MARKET_ROOT}/bounding_box_train/a.jpg")
    monkeypatch.setenv("ALLOW_TEST", "1")
    guard.check(f"{guard.MARKET_ROOT}/query/a.jpg")
    assert not (tmp_path / "flag").exists()


def test_market_rule_excludes_same_id_same_cam():
    feats = np.array([[1, 0], [1, 0], [0, 1], [0.9, 0.1]], np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    q = np.array([1, 0], np.float32)
    # 가장 가까운 두 장은 같은 ID·같은 카메라라 제외, 그다음이 정답(다른 카메라)이면 AP=1
    ap, r1 = evaluate_query(q, 7, 1, feats[[0, 1, 3, 2]], np.array([7, 7, 7, 8]), np.array([1, 1, 2, 3]))
    assert (ap, r1) == (1.0, 1.0)


def test_ap_value():
    q = np.array([1, 0], np.float32)
    g = np.array([[1, 0], [0.8, 0.6], [0.6, 0.8], [0, 1]], np.float32)
    ap, r1 = evaluate_query(q, 1, 1, g, np.array([2, 1, 3, 1]), np.array([5, 5, 5, 5]))
    assert r1 == 0.0
    assert ap == pytest.approx((1 / 2 + 2 / 4) / 2)


def test_no_valid_gallery_raises():
    q = np.array([1, 0], np.float32)
    with pytest.raises(ValueError):
        evaluate_query(q, 1, 1, np.array([[1, 0]], np.float32), np.array([1]), np.array([1]))


def test_pk_batches_shape():
    labels = np.repeat(np.arange(40), 6)
    labels[-2:] = 99
    batches = pk_batches(labels, 8, 4, np.random.default_rng(0))
    assert batches
    for b in batches:
        ids = labels[b]
        assert len(b) == 32 and len(set(ids)) == 8 and all((ids == i).sum() == 4 for i in set(ids))


def test_lr_schedule():
    o = {"lr": 1.0, "warmup_epochs": 10, "warmup_factor": 0.1, "milestones": [40, 70], "gamma": 0.1}
    assert lr_at(o, 0) == pytest.approx(0.1)
    assert lr_at(o, 10) == pytest.approx(1.0)
    assert lr_at(o, 40) == pytest.approx(0.1)
    assert lr_at(o, 70) == pytest.approx(0.01)


def test_bootstrap_verdict_and_mean():
    assert verdict(0.01, 0.2) == "useful"
    assert verdict(-0.2, -0.01) == "harmful"
    assert verdict(-0.1, 0.1) == "inconclusive"
    m = resample_means(np.array([2.0, 0.0]), np.array([2.0, 2.0]), np.array([[2.0, 0.0], [1.0, 1.0]]))
    assert m == pytest.approx([1.0, 0.5])
