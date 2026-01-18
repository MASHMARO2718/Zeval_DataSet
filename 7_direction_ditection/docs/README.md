# ドキュメント構成

このフォルダには、プロジェクトの各種ドキュメントが整理されています。

## 📁 フォルダ構成

```
docs/
├── paper/          # 論文執筆用ドキュメント
│   ├── TECHNICAL_REPORT_FOR_PAPER.md       # 技術詳細レポート（Methods/Results用）
│   ├── MEDIAPIPE_ERROR_ANALYSIS.md         # MediaPipe誤差の根本原因分析（Discussion用）
│   └── FINDINGS_AND_INSIGHTS.md            # 研究知見と洞察の総まとめ
│
└── archive/        # 古いドキュメント（参考用）
    ├── coordinate_transform_plan.md        # 座標変換の初期計画書
    ├── CURRENT_LOGIC.md                    # 初期の座標変換ロジック説明
    ├── IMPLEMENTATION_COMPLETE.md          # 実装完了報告
    ├── QUICKSTART.md                       # 古いクイックスタートガイド
    └── README_old.md                       # 古いREADME
```

---

## 📚 論文執筆用ドキュメント (`paper/`)

### 1. TECHNICAL_REPORT_FOR_PAPER.md（技術詳細レポート）

**用途**: 論文のMethods、Resultsセクション

**内容**:
- 評価指標の数学的定義（角度誤差Δθ、Δψ）
- Pearson相関係数の計算方法
- 座標系変換の詳細手順
- 定量的分析結果（関節ごとの統計）
- 高相関ペアの一覧表

**論文への活用**:
- 数式、表、実装コードをそのまま引用可能
- 再現性を確保するための詳細手順

---

### 2. MEDIAPIPE_ERROR_ANALYSIS.md（MediaPipe誤差の根本原因分析）

**用途**: 論文のDiscussionセクション

**内容**:
- MediaPipeの不正確性の根本原因（学習データの偏り60-70%、単眼カメラの限界25%、モデル制約15%）
- 各誤差パターンの詳細考察
  - 肘の±121°誤差 → 学習データの姿勢の偏り
  - 腰の左右対称誤差 → 単眼カメラの奥行き推定の限界
  - 足首・肩の高変動誤差 → 2Dキーポイント検出の不安定性
  - 上肢の連動誤差 → 階層的推定による誤差伝播
- 検証実験の提案
- **論文考察欄用テキスト**（2500字以上、そのままコピペ可能）

**論文への活用**:
- Discussionセクションの原因考察部分
- 学術的文体で記述済み

---

### 3. FINDINGS_AND_INSIGHTS.md（研究知見と洞察の総まとめ）

**用途**: 論文のIntroduction、Discussion、Conclusion

**内容**:
- 研究概要とデータセット詳細
- 座標系変換の技術的詳細
- MediaPipeの特性（長所と短所）
- 定量的分析結果のサマリー
- 相関分析の主要発見
- 今後の研究課題

**論文への活用**:
- 全体的な流れを掴むためのマスタードキュメント
- 各セクションへの参照元

---

## 🎓 論文執筆の推奨ワークフロー

### ステップ1: 全体像の把握
→ `FINDINGS_AND_INSIGHTS.md` を読む

### ステップ2: Methodsセクション執筆
→ `TECHNICAL_REPORT_FOR_PAPER.md` のセクション2, 3を引用

### ステップ3: Resultsセクション執筆
→ `TECHNICAL_REPORT_FOR_PAPER.md` のセクション4の表・統計を使用

### ステップ4: Discussionセクション執筆
→ `MEDIAPIPE_ERROR_ANALYSIS.md` の「論文考察欄用テキスト」をベースに執筆

### ステップ5: 図表の作成
→ `output/correlation_analysis/` のヒートマップを使用

---

## 📦 アーカイブドキュメント (`archive/`)

開発過程で作成された古いドキュメントです。現在は使用していませんが、開発の経緯を知りたい場合に参照できます。

| ファイル | 内容 | 状態 |
|---------|------|------|
| `coordinate_transform_plan.md` | 座標変換の初期計画書 | 実装完了済み |
| `CURRENT_LOGIC.md` | 初期のロジック説明 | 他のドキュメントに統合済み |
| `IMPLEMENTATION_COMPLETE.md` | 実装完了報告 | 参考情報 |
| `QUICKSTART.md` | 古いクイックスタート | README.mdに統合済み |
| `README_old.md` | 古いREADME | README.mdに更新済み |

---

## 📖 推奨閲覧順序

### 初めての方
1. プロジェクトルートの `README.md`（実行手順）
2. `paper/FINDINGS_AND_INSIGHTS.md`（研究の全体像）

### 論文執筆者
1. `paper/FINDINGS_AND_INSIGHTS.md`（全体像）
2. `paper/TECHNICAL_REPORT_FOR_PAPER.md`（Methods/Results）
3. `paper/MEDIAPIPE_ERROR_ANALYSIS.md`（Discussion）

### 開発者
1. プロジェクトルートの `README.md`
2. `archive/IMPLEMENTATION_COMPLETE.md`（実装の経緯）

---

**最終更新**: 2026年1月18日  
**ドキュメント管理**: プロジェクトチーム
