## 決定分析におけるPythonの利用
### 3.1 利得行列の作成手順

```
稼働させる機械の台数を決定するという問題を扱う
この意思決定の問題を整理する
  - 選択肢: 機械の稼働台数（0台・1台・2台）
  - 自然の状態: 製品の需要量（好況・不況）
  
ここで, 以下のパラメータを導入する
  - 工場の固定費用（fixed_cost）: 100万円
  - 機械の1台の稼働コスト（run_cost）: 600万円
  - 製品1つの販売価格（sale_price）: 0.2万円
  - 機械1台で作られる製品数（machine_ability）: 5000個
  - 好況時の需要量（demand_boom）: 10000個
  - 不況時の需要量（demand_slump）: 5000個
```

### 3.2 利得行列の作成

```python
import numpy as np
import pandas as pd

pd.set_option("display.unicode.east_asian_width", True)

# 工場の固定費用（万円）
fixed_cost = 100
# 機械1台数の稼働コスト（万円）
run_cost = 600
# 製品1つの販売価格（万円）
sale_price = 0.2

# 機械1台で作られる製品数（個）
machine_ability = 5000
# 好況時の需要量（個）
demand_boom = 10000
# 不況時の需要量（個）
demand_slump = 5000


def calc_payoff_table():
    # 出荷される製品の個数
    num_product_df = pd.DataFrame({
        "0台": [0, 0],
        "1台": [
                min([machine_ability, demand_boom]),
                min([machine_ability, demand_slump])
               ],
        "2台": [
                min([machine_ability * 2, demand_boom]),
                min([machine_ability * 2, demand_slump])
               ]
    })
    # 売上行列
    sales_df = num_product_df * sale_price
    # 製造コスト
    run_cost_df = pd.DataFrame({
        "0台": np.repeat(fixed_cost, 2),
        "1台": np.repeat(fixed_cost + run_cost, 2),
        "2台": np.repeat(fixed_cost + run_cost * 2, 2)
    })
    # 利得行列
    payoff_df = sales_df - run_cost_df
    payoff_df.index = ["好況", "不況"]
    return payoff_df
```

### 3.3 各基準の実装

```python
# マキシマックス基準
def argmax_list(series):
    return list(series[series == series.max()].index)


# マキシミン基準
def argmin_list(series):
    return list(series[series == series.min()].index)


# ハーヴィッツの基準
def hurwicz(payoff_table, alpha):
    hurwicz = payoff_table.max() * alpha + payoff_table.min() * (1 - alpha)
    return argmax_list(hurwicz)


# ミニマックスリグレット基準
def minimax_regret(payoff_table):
    best_df = pd.concat(
        [payoff_table.max(axis=1)] * payoff_table.shape[1], axis=1
    )
    best_df.columns = payoff_table.columns
    regret_df = best_df - payoff_table
    return argmin_list(regret_df.max())
```

### 3.4 感度分析

```
利得行列を計算する前提となった数値が少し変わるだけで, 意思決定の結果が大きく変わるようならば,
決定基準から得られた結果を採用するのは慎重になるべきかもしれない
感度分析は, 「モデルの前提となった数値の変化が, 意思決定の結果にどれほど影響を与えるか」を調べる作業である
```
