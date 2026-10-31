"""Path 1 기준선. 기본은 LR-only(보조 기준선), --mixed는 p(LR)=0.5 혼합(주 기준선). 등록 근거는 README."""
import argparse

from reid.trainer import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mixed", action="store_true")
    args = ap.parse_args()
    train(args.config, "mixed" if args.mixed else "lr_only")
