# GroundTruthからMediaPipe座標系への変換プラン（改訂版）

## 全体フロー

```
GroundTruth（Ybotワールド座標）
    ↓ Step 1: 座標系変換（右手系→左手系）
Left-Hand座標系 (x_LH, y_LH, z_LH)
    ↓ Step 2: 相対座標化（腰を原点とする）
相対座標 (x_stand, y_stand, z_stand)
    ↓ Step 3: MediaPipe出力との比較
差分ベクトルの計算
    ↓ Step 4: 角度計算
角度差θ（XY平面）, ψ（XZ平面）
    ↓ Step 5: 補正方向の決定
```

---

## Step 1: 座標系変換（右手系→左手系）

### GroundTruthの座標系（想定）
- X軸: 右方向
- Y軸: 上方向（Y-up）
- Z軸: 前方向
- 座標系: 右手座標系

### MediaPipeの座標系（Left-Hand系）
- X軸: 画像の左→右
- Y軸: 画像の上→下（Y-down）
- Z軸: カメラに近い→遠い
- 座標系: 左手座標系
- **原点**: 左腰（23番）と右腰（24番）の中点

### 変換式

GroundTruthの座標を $(x_{GT}, y_{GT}, z_{GT})$ とすると、Left-Hand座標系 $(x_{LH}, y_{LH}, z_{LH})$ への変換は：

```python
x_LH = x_GT
y_LH = -y_GT  # Y軸を反転（上下を反転）
z_LH = z_GT   # カメラ視点が同一のためそのまま
```

**前提条件:**
- GroundTruthとMediaPipeのカメラ視点が完全に同一
- Z軸の向きは変換不要

---

## Step 2: 相対座標化（腰を原点とする）

### MediaPipeの腰の定義

MediaPipeでは以下の関節点を使用：
- `LEFT_HIP` (23番): 左腰
- `RIGHT_HIP` (24番): 右腰

腰の中心座標（原点）：

```python
# GroundTruthデータから腰の中点を計算（LH座標系で）
hip_center_x = (left_hip_x_LH + right_hip_x_LH) / 2
hip_center_y = (left_hip_y_LH + right_hip_y_LH) / 2
hip_center_z = (left_hip_z_LH + right_hip_z_LH) / 2
```

### 各関節の相対座標化

各関節の座標から腰の中心座標を引く：

```python
x_stand = x_LH - hip_center_x
y_stand = y_LH - hip_center_y
z_stand = z_LH - hip_center_z
```

これで全ての関節が**腰を原点とした相対座標** $(x_{stand}, y_{stand}, z_{stand})$ になる。

**重要:** MediaPipeの出力も同じく腰が原点なので、この時点でスケールと原点が揃う。

---

## Step 3: MediaPipe出力との比較

### 前提条件
- **フレームレートが同一**: フレームごとに1対1で対応
- **欠損フレームは処理しない**: MediaPipeで検出失敗したフレームはスキップ

### 各関節ごとの差分ベクトル計算

GroundTruthの相対座標を $(x_{GT\_stand}, y_{GT\_stand}, z_{GT\_stand})$、MediaPipe出力を $(x_{MP}, y_{MP}, z_{MP})$ とすると：

```python
# 差分ベクトル（3次元）
Δx = x_MP - x_GT_stand
Δy = y_MP - y_GT_stand
Δz = z_MP - z_GT_stand

# ユークリッド距離による誤差
error = sqrt(Δx² + Δy² + Δz²)
```

**注意:** 角度計算にはスケールは不要だが、誤差の大きさを評価する場合は正規化を検討。

---

## Step 4: 角度差の計算

### 角度の数学的性質

**重要:** 角度は**方向（向き）**のみで決まり、スケールに依存しない。

```
θ = atan2(y, x)  # xとyの比率だけで決まる
ψ = atan2(z, x)  # xとzの比率だけで決まる
```

したがって、**正規化なしで直接角度を計算できる**。

### 角度θ（XY平面での角度）

XY平面に投影した場合の角度：

```python
# GroundTruthの角度（腰からの向き）
θ_GT = atan2(y_GT_stand, x_GT_stand)

# MediaPipe出力の角度
θ_MP = atan2(y_MP, x_MP)

# 角度差
Δθ = θ_MP - θ_GT

# -πからπの範囲に正規化
Δθ = normalize_angle(Δθ)
```

### 角度ψ（XZ平面での角度）

XZ平面に投影した場合の角度：

```python
# GroundTruthの角度（腰からの向き）
ψ_GT = atan2(z_GT_stand, x_GT_stand)

# MediaPipe出力の角度
ψ_MP = atan2(z_MP, x_MP)

# 角度差
Δψ = ψ_MP - ψ_GT

# -πからπの範囲に正規化
Δψ = normalize_angle(Δψ)
```

### 角度の正規化関数

```python
import numpy as np

def normalize_angle(angle):
    """角度を-πからπの範囲に正規化"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle
```

---

## Step 5: 補正方向の決定

### 補正ベクトルの計算

角度差から、どの方向に補正すべきかを計算：

#### XY平面での補正

```python
# MediaPipeの位置ベクトルの大きさ（XY平面）
r_xy_MP = sqrt(x_MP² + y_MP²)

# GroundTruthの方向ベクトル（単位ベクトル）
direction_GT_xy = [cos(θ_GT), sin(θ_GT)]

# MediaPipeの方向ベクトル（単位ベクトル）
direction_MP_xy = [cos(θ_MP), sin(θ_MP)]

# 補正方向ベクトル（MediaPipe→GroundTruth）
correction_xy = [
    r_xy_MP * (cos(θ_GT) - cos(θ_MP)),
    r_xy_MP * (sin(θ_GT) - sin(θ_MP))
]
```

#### XZ平面での補正

```python
# MediaPipeの位置ベクトルの大きさ（XZ平面）
r_xz_MP = sqrt(x_MP² + z_MP²)

# GroundTruthの方向ベクトル（単位ベクトル）
direction_GT_xz = [cos(ψ_GT), sin(ψ_GT)]

# MediaPipeの方向ベクトル（単位ベクトル）
direction_MP_xz = [cos(ψ_MP), sin(ψ_MP)]

# 補正方向ベクトル（MediaPipe→GroundTruth）
correction_xz = [
    r_xz_MP * (cos(ψ_GT) - cos(ψ_MP)),
    r_xz_MP * (sin(ψ_GT) - sin(ψ_MP))
]
```

---

## 座標系変換の検証方法

### 方法1: 3Dプロットによる視覚確認

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1フレーム分のデータをプロット
fig = plt.figure(figsize=(12, 6))

# GroundTruth
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(x_GT_stand, y_GT_stand, z_GT_stand, c='blue', label='GroundTruth')
ax1.set_title('GroundTruth (相対座標)')

# MediaPipe出力
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(x_MP, y_MP, z_MP, c='red', label='MediaPipe')
ax2.set_title('MediaPipe Output')

plt.show()
```

### 方法2: 特定関節の軌跡比較

```python
# 例: 右手首の軌跡を時系列でプロット
plt.figure(figsize=(10, 4))
plt.plot(frames, right_wrist_x_GT, label='GT X', linestyle='--')
plt.plot(frames, right_wrist_x_MP, label='MP X', linestyle='-')
plt.legend()
plt.title('Right Wrist X Coordinate Over Time')
plt.show()
```

### 方法3: 既知の身体比率の確認

```python
# 肩幅の比較（左右対称性の確認）
shoulder_width_GT = distance(left_shoulder_GT, right_shoulder_GT)
shoulder_width_MP = distance(left_shoulder_MP, right_shoulder_MP)

print(f"GroundTruth肩幅: {shoulder_width_GT:.3f}")
print(f"MediaPipe肩幅: {shoulder_width_MP:.3f}")
print(f"比率: {shoulder_width_MP / shoulder_width_GT:.3f}")
```

---

## 実装の流れ（推奨）

### Phase 1: 座標変換の検証
1. 1フレーム分のデータで座標変換を実施
2. 3Dプロットで視覚的に確認
3. 腰が原点(0,0,0)になっているか確認

### Phase 2: 全フレーム処理
1. 全フレームで座標変換と相対座標化
2. 欠損フレームをスキップ
3. 結果をCSVまたはNumPy配列で保存

### Phase 3: 角度計算
1. 各関節・各フレームでθとψを計算
2. 角度差の統計量を計算（平均、標準偏差など）
3. 誤差の大きい関節を特定

### Phase 4: 可視化と分析
1. 角度差のヒートマップ作成
2. 時系列での角度差の変化を確認
3. 補正フィルタの設計へ

---

## データ保存形式

```csv
frame_id, joint_id, joint_name, x_GT_stand, y_GT_stand, z_GT_stand, x_MP, y_MP, z_MP, theta_GT, theta_MP, delta_theta, psi_GT, psi_MP, delta_psi, error_3d
0, 11, LEFT_SHOULDER, 0.123, 0.456, 0.789, 0.120, 0.450, 0.780, 1.234, 1.240, 0.006, 0.567, 0.570, 0.003, 0.012
...
```

---

## まとめ：重要なポイント

✅ **スケール正規化は不要**: 角度は方向のみで決まる  
✅ **腰が原点**: GroundTruthもMediaPipeも腰の中点を原点とする  
✅ **Y軸を反転**: 右手系→左手系の変換で `y_LH = -y_GT`  
✅ **フレーム同期済み**: 1対1対応、欠損フレームはスキップ  
✅ **検証が重要**: 3Dプロットで目視確認してから全フレーム処理
