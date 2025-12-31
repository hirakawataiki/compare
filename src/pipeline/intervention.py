# 盛り上がりスコアの変化と沈黙時間に基づく介入判定ロジック
# 直近の盛り上がりスコアと基準スコアを比較し、一定時間の沈黙が続いた場合に介入を判定する
# 条件：沈黙が一定秒以上 ＋ スコアが基準の一定割合以下

from dataclasses import dataclass

@dataclass
class InterventionSignal:
    trigger: bool
    reason: str

def should_intervene(silence_sec: float, current_score: float, baseline: float,
                     silence_th=3.5, drop_ratio_th=0.6) -> InterventionSignal:
    # 直近スコアが基準の drop_ratio_th 以下 & 沈黙が一定秒を超えたら介入
    drop = (baseline > 0) and (current_score <= baseline * drop_ratio_th)
    if silence_sec >= silence_th and drop:
        return InterventionSignal(True, f"Silence {silence_sec:.1f}s & score drop {current_score:.2f}/{baseline:.2f}")
    return InterventionSignal(False, "")
