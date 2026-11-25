"""Bước 3: huấn luyện + đánh giá một nhóm thực nghiệm (configs/experiments/<tên>.yaml).

Ví dụ:
  python scripts/03_run.py smoke                       # chạy thử toàn pipeline
  python scripts/03_run.py main_test                   # 4 mô hình x 5 seed
  python scripts/03_run.py main_test --only unet --seeds 101 202
  python scripts/03_run.py context_sweep --set train.num_workers=4
  python scripts/03_run.py main_test --eval-only       # chỉ đánh giá lại từ checkpoint
Lần chạy đã có summary.json sẽ được bỏ qua -> chạy lại lệnh để tiếp tục sau khi Kaggle ngắt phiên.
"""
import argparse

import _bootstrap  # noqa: F401
from acacia.runner import run_experiment

ap = argparse.ArgumentParser()
ap.add_argument("experiment")
ap.add_argument("--only", nargs="*")
ap.add_argument("--seeds", nargs="*", type=int)
ap.add_argument("--device")
ap.add_argument("--eval-only", action="store_true")
ap.add_argument("--force", action="store_true", help="huấn luyện lại dù đã xong")
ap.add_argument("--set", nargs="*", default=[], help="ghi đè config, vd train.max_epochs=5")
a = ap.parse_args()
run_experiment(a.experiment, a.seeds, a.only, a.device, a.set, a.eval_only, a.force)
