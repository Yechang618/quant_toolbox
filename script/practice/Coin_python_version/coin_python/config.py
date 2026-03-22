import os
import sys

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    OKX_HIS_DATA_BASE_PATH = '' # your saved path
