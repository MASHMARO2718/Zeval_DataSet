import pandas as pd
import numpy as np

# 2つのCSVファイルを読み込む
df1 = pd.read_csv('coordinate_angle_mae_Y=0.5,1.5.csv')
df2 = pd.read_csv('coordinate_angle_mae_Y=1.0,2.0.csv')

# データを統合
df_combined = pd.concat([df1, df2], ignore_index=True)

print(f"統合後のデータ数: {len(df_combined)} 行")
print(f"統合前のデータ数: {len(df1)} + {len(df2)} = {len(df1) + len(df2)} 行")

# 統合したデータをCSVとして保存
df_combined.to_csv('coordinate_angle_mae_combined.csv', index=False)
print(f"\n統合データを 'coordinate_angle_mae_combined.csv' に保存しました。\n")

# 関節のリスト（肩、肘、腰、膝すべて）
joints = {
    'L_Shoulder': '左肩',
    'R_Shoulder': '右肩',
    'L_Elbow': '左肘',
    'R_Elbow': '右肘',
    'L_Hip': '左腰',
    'R_Hip': '右腰',
    'L_Knee': '左膝',
    'R_Knee': '右膝'
}

# 統計を計算
results = []

for joint_col, joint_name in joints.items():
    # 空の値（NaN）を除外
    values = df_combined[joint_col].dropna()
    
    if len(values) > 0:
        mae = values.mean()
        median = values.median()
        max_error = values.max()
        
        results.append({
            '関節': joint_name,
            '平均誤差 (度)': f'{mae:.1f}',
            '中央値 (度)': f'{median:.1f}',
            '最大誤差 (度)': f'{max_error:.1f}'
        })
        
        print(f"{joint_name}:")
        print(f"  平均誤差: {mae:.1f}度")
        print(f"  中央値: {median:.1f}度")
        print(f"  最大誤差: {max_error:.1f}度")
        print(f"  データ数: {len(values)}")
        print()

# テーブルを作成
df_table = pd.DataFrame(results)

# テーブルを表示
print("=" * 60)
print("Table 1: 関節角度誤差の統計")
print("=" * 60)
print(df_table.to_string(index=False))
print("=" * 60)

# CSVとして保存
df_table.to_csv('joint_angle_error_statistics.csv', index=False, encoding='utf-8-sig')
print(f"\n統計テーブルを 'joint_angle_error_statistics.csv' に保存しました。")

# LaTeX形式でも保存（オプション）
latex_table = df_table.to_latex(index=False, escape=False)
with open('joint_angle_error_statistics.tex', 'w', encoding='utf-8') as f:
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{関節角度誤差の統計}\n")
    f.write("\\label{tab:joint_angle_error}\n")
    f.write(latex_table)
    f.write("\\end{table}\n")
print("LaTeX形式のテーブルを 'joint_angle_error_statistics.tex' に保存しました。")

