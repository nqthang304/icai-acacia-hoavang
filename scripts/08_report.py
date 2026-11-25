"""Bước 8: tổng hợp toàn bộ bảng + hình vào work/results/report/ (REPORT.md, *.csv, *.png).
Chạy được bất cứ lúc nào: phần nào chưa có kết quả sẽ được bỏ qua.
"""
import _bootstrap  # noqa: F401
from acacia.config import load_config
from acacia.data.context import DataContext
from acacia.report.build import Report

cfg = load_config()
r = Report(cfg)
r.main_tables()
r.validation_tables()
r.dropout_table()
r.polygon_tables()
r.training_table()
r.prediction_maps(DataContext(cfg))
print("->", r.write())
