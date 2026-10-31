"""train651 HR로 ReID를 학습한다. 이 모델은 모든 SR 방법 평가에 고정해서 쓴다."""
import argparse

from reid.trainer import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    train(ap.parse_args().config, "hr")
