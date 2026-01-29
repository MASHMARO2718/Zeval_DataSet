"""
相関行列（ψ）の検証スクリプト
ヒートマップの「青い帯」が純粋な結果か、計算ミスかを切り分けるため、
生データから相関を再計算し、保存済み行列と突き合わせる。
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    base = Path(__file__).resolve().parent.parent
    detailed_csv = base / 'output' / 'processed_data' / 'detailed_results.csv'
    saved_csv = base / 'output' / 'correlation_analysis' / 'correlation_matrix_psi.csv'

    if not detailed_csv.exists():
        print(f"[ERROR] Not found: {detailed_csv}")
        return
    if not saved_csv.exists():
        print(f"[ERROR] Not found: {saved_csv}")
        return

    print("=" * 60)
    print("  相関行列（ψ）検証：青い帯は本当のパターンか？")
    print("=" * 60)

    # 1) 生データから相関行列を再計算（compute_correlation.py と同じロジック）
    df = pd.read_csv(detailed_csv)
    pivot = df.pivot_table(
        index=['frame_id', 'camera'],
        columns='joint',
        values='delta_psi_deg',
        aggfunc='first'
    )
    pivot_clean = pivot.dropna()
    corr_computed = pivot_clean.corr(method='pearson')

    # 2) 対角・対称性チェック
    diag_ok = np.allclose(np.diag(corr_computed.values), 1.0)
    sym_ok = np.allclose(corr_computed.values, corr_computed.values.T)
    print(f"\n[チェック] 対角がすべて 1.0: {diag_ok}")
    print(f"[チェック] 行列が対称: {sym_ok}")

    # 3) LEFT_HIP vs RIGHT_HIP を手計算（np.corrcoef）で検証
    col_l = pivot_clean['LEFT_HIP'].values
    col_r = pivot_clean['RIGHT_HIP'].values
    r_hand = np.corrcoef(col_l, col_r)[0, 1]
    r_from_matrix = corr_computed.loc['LEFT_HIP', 'RIGHT_HIP']
    print(f"\n[LEFT_HIP vs RIGHT_HIP]")
    print(f"  相関行列から:     {r_from_matrix:.6f}")
    print(f"  np.corrcoef から: {r_hand:.6f}")
    print(f"  一致: {np.isclose(r_from_matrix, r_hand)}")

    # 4) 保存済み CSV と比較
    saved = pd.read_csv(saved_csv, index_col=0)
    # 列順が違う可能性があるので、共通の関節で比較
    common = [c for c in corr_computed.columns if c in saved.columns]
    diff = (corr_computed.loc[common, common] - saved.loc[common, common]).abs().max().max()
    print(f"\n[保存済み CSV との差] 最大絶対差: {diff:.6f}")
    r_saved = saved.loc['LEFT_HIP', 'RIGHT_HIP']
    print(f"  保存 CSV の LEFT_HIP-RIGHT_HIP: {r_saved:.6f}")

    # 5) 「青い帯」の正体：どの関節の行がほぼすべて負か
    print("\n[青い帯の正体] 他関節との相関がすべて負になる関節（行）:")
    for j in corr_computed.index:
        row = corr_computed.loc[j].drop(j)  # 自分自身を除く
        if (row < 0).all():
            print(f"  → {j}: 他 {len(row)} 関節すべてと負の相関（青い横線＋縦線）")

    print("\n" + "=" * 60)
    if diag_ok and sym_ok and np.isclose(r_from_matrix, r_hand) and diff < 1e-5:
        print("結論: 計算・保存ともに一致。青い帯は「腰（LEFT_HIP 等）が他関節と負の相関を持つ」というデータ上のパターンであり、実装ミスではない。")
    else:
        print("結論: 不一致あり。要確認。")
    print("=" * 60)

if __name__ == '__main__':
    main()
