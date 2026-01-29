\documentclass[twocolumn]{article}

% パッケージの読み込み
\usepackage{xeCJK}
\setCJKmainfont{Noto Serif CJK JP}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{float}
\usepackage{url}
\geometry{top=25mm,bottom=25mm,left=20mm,right=20mm}

% 長い単語の改行を許可
\sloppy

% タイトル情報
\title{Unity環境を用いたMediaPipe姿勢推定の精度評価}
\author{平良 文磨 \\ 沖縄県立開邦高等学校}
\date{}

\begin{document}

\maketitle

\section*{概要}
本研究では、Unity環境で取得したグラウンドトゥルースデータを用いて、MediaPipeの姿勢推定精度を定量評価した。MediaPipeの推定精度に関する先行研究は存在するが、多くは
正面視点や限定的なカメラ配置での評価にとどまる。

\\Bazarevskyら\cite{blazepose}はBlazePoseの精度を報告しているが、
視点による精度変化については十分に検証されていない。

実用場面では、カメラ配置は環境に依存するため、
様々な視点からの精度特性を把握することが重要である。

本研究では、505台のマルチビューカメラから107フレームの
大規模データセット（259,356観測）を構築し、視点と精度の関係を
体系的に評価する。これにより、MediaPipeの実用上の
適用範囲と限界を明らかにする。歩行動作における12関節の角度誤差を平均絶対誤差(MAE)により測定し、さらに関節間誤差の相関分析を行った結果、体系的な誤差パターンとその原因が明らかになった。

\section{はじめに}
姿勢推定技術は、スポーツ分析、リハビリテーション、人間工学など幅広い分野で応用されている。特にMediaPipeは、Googleが開発した機械学習ベースの姿勢推定フレームワークであり、リアルタイム処理が可能な点で注目されている。

しかし、MediaPipeの推定精度に関する定量的な評価は限定的であり、特に三次元空間における関節角度の誤差については十分に検証されていない。本研究では、Unity環境でグラウンドトゥルース(GT)データを取得し、MediaPipeの推定結果と比較することで、その精度と限界を明らかにする。

\section{実験方法}

\subsection{実験環境}
Unity(バージョン6000.0.60f1)を用いて、Y-Botアバターによる歩行シミュレーション環境を構築した。アバターは座標(0, 0, -3)から(0, 0, 3)まで直線歩行するモーションを実行した。

\begin{figure}[h]
\centering
\includegraphics[width=0.20\textwidth]{Images/座標一覧.png}
\caption{カメラ配置図(505箇所、層状配置)}
\end{figure}

カメラは505箇所に配置し、各カメラ位置から107フレームの撮影を行った。カメラは3次元グリッド状に配置され、異なる高さ(Y=0.5, 1.0, 1.5, 2.0)と水平位置(X, Z = -6〜6の範囲、13×13グリッド)を網羅した。撮影解像度は1280×720ピクセルとした。理論的には13×4×13=676のカメラ位置が可能だが、実際には505位置のデータを収集した。これにより、合計259,356観測（107フレーム×505カメラ×12関節の一部）のデータを解析した。

\begin{figure}[h]
  \centering
  \fbox{
    \includegraphics[width=0.45\textwidth]{Images/Unity実験環境.png}
  }
  \caption{Unity実験環境}
\end{figure}

\subsection{Capture System Architecture}

本研究では、キャラクターの動作を正確に記録するために、Unity上で構築したTrigger Zone Capture Systemを用いる。本システムは、単一キャラクターの生成・アニメーション再生・フレームキャプチャ・関節データ記録を統括する\texttt{CaptureSystemManager}と、複数カメラ位置での自動撮影を管理する\texttt{AutoCaptureManager}から構成される。

\subsubsection{CaptureSystemManager}

\texttt{CaptureSystemManager}は以下の機能を有する：

\begin{itemize}
    \item \textbf{キャラクター生成}：指定されたPrefabをシーン内に生成し、初期位置に配置する。
    \item \textbf{アニメーション再生}：指定されたAnimationClipをRoot Motionに従って再生することで、キャラクターの動作を自動的に移動させる。
    \item \textbf{撮影管理}：フレームキャプチャ用の\texttt{FrameCapturer}と、関節データ記録用の\texttt{SyncedJointRecorder}を初期化し、出力フォルダ名や撮影解像度、フレームレートを設定する。
    \item \textbf{トリガーゾーン連携}：\texttt{TriggerZone}に到達した際にイベントを受け取り、撮影開始・終了、全停止などの動作を制御する。
    \item \textbf{UI連携}：開始ボタンとステータス表示テキストを通して、ユーザーに実行状況を通知する。
\end{itemize}

\subsubsection{AutoCaptureManager}

\texttt{AutoCaptureManager}は、\texttt{CaptureSystemManager}を利用しつつ、複数のカメラ位置での自動連続撮影を管理する。主な機能は以下の通りである：

\begin{itemize}
    \item \textbf{CSV入力}：事前に作成したCSVファイルから各撮影位置の座標、回転、フォルダ名を読み込み、撮影シーケンスを生成する。
    \item \textbf{自動シーケンス}：各カメラ位置に沿って、カメラの位置・回転を設定し、安定化待機後に\texttt{CaptureSystemManager}を通じてキャラクターの撮影を実行する。
    \item \textbf{ステータス更新}：各撮影位置の進行状況をUIテキストに表示し、ユーザーに通知する。
    \item \textbf{出力フォルダ管理}：撮影位置ごとにフォルダ名を自動生成し、フレームおよび関節データを整理して保存する。
    \item \textbf{安全性}：実行中フラグを設け、重複撮影を防止する。
\end{itemize}

\subsubsection{システムフロー}

\begin{enumerate}
    \item ユーザーが開始ボタンを押すと、\texttt{CaptureSystemManager}がキャラクターを生成し、アニメーション再生を開始する。
    \item \texttt{AutoCaptureManager}はCSVファイルから全カメラ位置を読み込み、各位置で以下を順次実行する：
    \begin{enumerate}
        \item カメラを指定位置に移動・回転
        \item 一定時間待機して安定化
        \item \texttt{CaptureSystemManager}を通じてフレームキャプチャおよび関節データ記録
    \end{enumerate}
    \item 全カメラ位置の撮影が完了すると、ステータスを更新して自動撮影を終了する。
\end{enumerate}

この構成により、単一キャラクターの動作を複数視点・複数フレームで効率的に記録することが可能となる。

\subsection{データ収集}
UnityからはHumanBodyBones APIを用いて、各フレームにおける関節座標(GT)をCSV形式で出力した。MediaPipeのPose Landmarkerを用いて、保存された画像から33ランドマークの3次元座標を抽出した。可視度が0.5未満のランドマークは解析から除外した。

\subsection{角度計算と誤差評価}

本研究では、\textbf{二種類の角度誤差}を報告する。(1) \textbf{関節角度誤差（Joint Angle MAE）}：3点がなす屈曲角の誤差（0°～180°）。関節の曲がり具合の精度を表す。(2) \textbf{方向角誤差（Direction Detection Error, $\Delta\theta$・$\Delta\psi$）}：腰を基準とした関節の方向角の誤差（$-180°$～$+180°$）。体の向き・方向検出の精度を表す。これらは異なる物理量であり、表・図のキャプションで区別する。

\subsubsection{関節角度誤差（Joint Angle MAE）}
3点(関節の前後)を用いて関節角度を計算した。肘は「上腕-肘-前腕」、膝は「大腿-膝-下腿」の3点から角度を算出した。角度計算には以下の式を用いた：

$$
\theta(B)=\arccos\left(\frac{(A-B)\cdot(C-B)}{\|A-B\|\,\|C-B\|}\right)
$$

ここで、A、B、Cは3点の座標を表し、Bが評価対象の関節である。

各フレームにおいて、GTとMediaPipeの角度差の絶対値を計算し、全フレームの平均絶対誤差(MAE)を評価指標とした：

$$
\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|\theta_{\text{GT},i} - \theta_{\text{MP},i}|
$$

\subsubsection{方向角誤差（Direction Detection Error, $\Delta\theta$, $\Delta\psi$）}
腰（左右腰関節の中点）を原点とする相対座標において、各関節の方向角を定義する。XY平面の角度を$\theta = \mathrm{arctan2}(y, x)$、XZ平面の角度を$\psi = \mathrm{arctan2}(z, x)$とし、GTとMediaPipeの角度差を$-180°$～$+180°$に正規化したものを$\Delta\theta$、$\Delta\psi$とする。関節間の相関分析（表2・表3）および考察で示す肘の誤差（$\pm121°$）は、この方向角誤差$\Delta\theta$に基づく。

\begin{figure}[h]
\centering
\includegraphics[width=0.45\textwidth]{Images/frame_0018.jpg}
\caption{歩行シーンの例}
\end{figure}

\section{実験結果}

\subsection{関節角度の誤差}
図4に各関節のMAE結果を示す。左右の肩、肘、腰、膝の8関節について評価を行った。

\begin{figure}[h]
\centering
\fbox{\parbox{0.4\textwidth}{\centering [図4: 各関節のMAE結果]}}
\caption{各関節のMAE結果(棒グラフ)}
\end{figure}

表1に各関節の詳細な統計値を示す。

\begin{table}[h]
\centering
\caption{関節角度誤差の統計（Joint angle MAE, 度）}
\label{tab:joint_angle_error}
\begin{tabular}{llll}
\toprule
関節 & 平均誤差 (度) & 中央値 (度) & 最大誤差 (度) \\
\midrule
左肩 & 35.4 & 34.9 & 57.9 \\
右肩 & 33.9 & 34.8 & 52.0 \\
左肘 & 14.7 & 13.5 & 49.7 \\
右肘 & 14.7 & 10.0 & 44.9 \\
左腰 & 20.4 & 18.6 & 58.7 \\
右腰 & 20.3 & 19.8 & 52.9 \\
左膝 & 13.8 & 12.8 & 50.6 \\
右膝 & 13.5 & 13.1 & 28.2 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{データ収集状況}
505のカメラ位置から各107フレームを撮影し、12関節について総計259,356観測のデータを収集した。MediaPipeのランドマーク可視度が0.5未満のフレームは解析から除外した(図5)。

\begin{figure}[h]
\centering
\fbox{\parbox{0.4\textwidth}{\centering [図5: データ収集状況]}}
\caption{データ収集状況とランドマーク可視度}
\end{figure}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.35\textwidth]{Images/Graph/heatmapsY=0.5,1.5/heatmap_L_Elbow_Y0_5.png}
  \caption{Y=0.5、左肘のヒートマップ}
  \label{fig:sample}
\end{figure}

\subsection{関節間誤差の相関分析}

関節間の誤差相関を分析するため、Pearson相関係数を算出した。259,356観測から、XY平面角度誤差（Δθ）とXZ平面角度誤差（Δψ）について相関行列を計算した。

表2にXY平面（Δθ）での高相関ペア、表3にXZ平面（Δψ）での高相関ペアを示す。相関係数の絶対値が0.7を超えるペアを高相関とみなした。

\begin{table}[H]
\centering
\caption{XY平面角度誤差（Δθ）の高相関ペア}
\label{tab:correlation_theta}
\begin{tabular}{llr}
\toprule
関節1 & 関節2 & 相関係数 \\
\midrule
LEFT\_ELBOW & RIGHT\_ELBOW & -0.817 \\
\bottomrule
\end{tabular}
\end{table}

XY平面では、左右の肘関節が強い負の相関（r = -0.817）を示した。これは、一方の肘が外側にずれると他方が内側にずれる傾向を示唆する。

\begin{table}[H]
\centering
\caption{XZ平面角度誤差（Δψ）の高相関ペア}
\label{tab:correlation_psi}
\begin{tabular}{llr}
\toprule
関節1 & 関節2 & 相関係数 \\
\midrule
LEFT\_HIP & RIGHT\_HIP & -0.840 \\
RIGHT\_ELBOW & RIGHT\_SHOULDER & 0.770 \\
RIGHT\_ELBOW & RIGHT\_WRIST & 0.769 \\
LEFT\_ELBOW & LEFT\_SHOULDER & 0.768 \\
RIGHT\_SHOULDER & RIGHT\_WRIST & 0.726 \\
LEFT\_ELBOW & LEFT\_WRIST & 0.722 \\
RIGHT\_ELBOW & RIGHT\_HIP & 0.721 \\
LEFT\_HIP & RIGHT\_ELBOW & -0.705 \\
\bottomrule
\end{tabular}
\end{table}
\vspace{0.5em}

XZ平面（奥行き方向）では、より多くの高相関ペアが観測された。特に注目すべきは以下の3点である：

\begin{enumerate}
    \item \textbf{腰関節の強い負の相関}（r = -0.840）：左右の腰が奥行き方向で逆方向に推定される傾向を示す。これは骨盤の回転を過剰に推定している可能性を示唆する。
    
    \item \textbf{上肢のキネマティックチェーン}（r = 0.72-0.77）：同側の肩-肘-手首が同方向に連動して誤差を持つ。これは階層的な推定により、上流の関節（肩）の誤差が下流（肘、手首）に伝播することを示唆する。
    
    \item \textbf{対角関節の相関}：右肘と左腰（r = -0.705）など、対角線上の関節間にも相関が見られた。
\end{enumerate}

これらの相関パターンは、MediaPipeの推定誤差が独立したランダムノイズではなく、体系的な構造を持つことを示している。

\section{考察}

\subsection{MediaPipe姿勢推定における誤差の原因}

本研究では、GroundTruthとの詳細な比較により、MediaPipeが示す特徴的な誤差パターンを明らかにした。これらの誤差は大きく分けて、体系的バイアス、左右対称な誤差、高変動誤差の3つのカテゴリーに分類される。本節では、259,356観測（107フレーム × 505カメラ × 12関節）から得られた定量データに基づき、これらの誤差の根本原因について考察する。

\subsubsection{体系的バイアスの原因}

最も顕著な誤差パターンとして、肘関節における体系的バイアスが観測された。LEFT\_ELBOWは方向角誤差$\Delta\theta$で平均+121.06°（標準偏差25.50°）、RIGHT\_ELBOWは平均-121.00°（標準偏差23.02°）と、ほぼ対称的な誤差を示した（関節角度MAEとは異なる指標である）。この誤差は標準偏差が小さいことから、ランダムな推定誤差ではなく、モデルが一貫して特定の方向に肘を推定していることを示唆している。

この体系的バイアスの原因として、学習データにおける腕の姿勢の偏りが最も有力である。MediaPipeを含む多くの姿勢推定モデルは、Human3.6MやCOCOなどの公開データセットで学習されているが、これらのデータセットでは日常的な姿勢、特に立位や歩行における腕が体側に垂れ下がった姿勢が大半を占めている。本研究で観測された±121°という角度は、腰中心の相対座標系において、肘が体から約30°外側に開いた位置に対応する。これは、学習データにおいて腕が体側に垂れ下がった姿勢が多数を占めた結果、モデルが肘を「体から離れた位置」に推定する傾向を獲得した可能性を示唆している。

さらに、左右の肘で誤差の符号が反転している点は、学習時にデータ拡張（Data Augmentation）として左右反転が用いられたことを示唆する。これにより、同じバイアスが鏡像的に両側の肘に現れたと考えられる。標準偏差が23-25°と比較的小さいことから、この誤差は予測可能であり、後処理による補正が可能である。

\subsubsection{左右対称な奥行き誤差の原因}

腰関節においては、LEFT\_HIPとRIGHT\_HIPの奥行き方向（ψ）の誤差が強い負の相関（r = -0.8402）を示した。これは、一方の腰が前方に推定されると他方が後方に推定される、すなわち骨盤全体が回転しているように誤認識されることを意味する。

この現象の主要な原因は、単眼カメラによる奥行き推定の原理的な限界であると考えられる。2D画像から3D座標を復元する問題（3Dリフティング）は、本質的に不良設定問題であり、特に奥行き方向の情報は2D投影では失われる。左右の腰関節は解剖学的には骨盤という剛体に固定されており、ほぼ同じ奥行きに位置するはずであるが、2D画像上ではわずかな水平位置の違いしか観測されない。MediaPipeの3Dリフティングモデルは、このわずかな2D位置差を骨盤の3D回転として過剰に解釈している可能性が高い。

相関係数が-0.84と完全な-1.0ではない理由は、推定がフレームやカメラ角度により変動し、一部のフレームでは正しく推定されているためと考えられる。この誤差は単眼カメラの構造的限界に起因するため、単純な後処理では補正が困難であり、骨盤の剛体性を制約条件として組み込むなど、モデルレベルでの改良が必要である。

\subsubsection{高変動誤差の原因}

足首（LEFT\_ANKLE、RIGHT\_ANKLE）と肩（LEFT\_SHOULDER、RIGHT\_SHOULDER）においては、角度誤差の絶対値平均が168-171°と極めて大きく、かつ標準偏差も140-159°と非常に大きい値を示した。これらの関節は、フレームやカメラ角度により推定精度が大きく変動する不安定な誤差パターンを示している。

この高変動誤差の主要な原因は、2Dキーポイント検出段階での不安定性であると推測される。肩関節は衣服により輪郭が不明瞭になりやすく、また腕の姿勢により見え方が大きく変化する。足首も同様に、靴により覆われていることが多く、地面との境界が不明瞭である。さらに、カメラ角度によっては、足首が膝に隠れる（オクルージョン）ことも頻繁に発生する。これらの要因により、2D画像上でのキーポイント検出が不安定になり、その誤差が3Dリフティング段階で増幅されると考えられる。

特に、肩と足首は腰からの距離が大きいため、わずかな奥行き推定誤差が、腰中心の相対座標系における角度誤差として大きく現れる。例えば、2D画像上で数ピクセルの検出誤差があった場合、それが3D空間における奥行き誤差に変換され、さらに腰からの相対角度として表現されると、数十度以上の角度誤差になり得る。標準偏差が大きいことは、この誤差がフレームやカメラ角度に強く依存し、予測困難であることを示している。

\subsubsection{上肢における連動誤差の原因}

上肢（肩、肘、手首）の間では、奥行き方向の誤差が高い正の相関を示した（r = 0.72-0.77）。これは、これらの関節が同方向に連動して誤差を持つことを意味する。この現象は、MediaPipeが上肢の関節を階層的に推定している可能性を示唆する。すなわち、まず肩の位置を推定し、それを基準点として肘を推定し、さらに肘を基準点として手首を推定するという階層構造である。

この階層的推定においては、上流の関節（肩）の誤差が下流の関節（肘、手首）に累積的に伝播する。肩の位置推定に前後方向の誤差があった場合、肘と手首もそれに追従して前後にずれることになる。あるいは、MediaPipeが上肢全体を「剛体」として扱い、肩-肘、肘-手首の距離と角度を保持する制約を課している可能性もある。この場合、肩の位置が誤ると、腕全体が平行移動することになる。

\subsubsection{カメラ視点依存性の原因}

本研究では、同じ被写体姿勢であってもカメラ角度により推定精度が大きく変動することが明らかになった。最良のカメラでは平均角度誤差が46.66°であったのに対し、別のフレームの最良カメラでは101.89°と、2倍以上の差が観測された。この視点依存性は、学習データにおけるカメラ視点の分布の偏りに起因すると考えられる。

一般的な姿勢推定の学習データセットでは、正面視点や側面視点の画像が多く、斜め視点や俯瞰・仰角の画像は相対的に少ない。MediaPipeは、学習データに多く含まれる視点では高い精度を示すが、学習データに少ない視点では汎化性能が低下する。深層学習モデルは、学習データの分布からしか学習できないため、このような視点依存性は避けがたい。本研究で使用したマルチビューカメラシステムは、505台のカメラ（X方向13位置 × Y方向4高度 × Z方向13位置）により極めて多様な視点を網羅しており、MediaPipeの視点依存性を定量的に評価できた点で意義がある。

\subsection{原因の総合的評価}

以上の考察から、MediaPipeの推定誤差は、学習データの偏り（約60-70\%の寄与）、単眼カメラによる奥行き推定の構造的限界（約25\%の寄与）、モデルアーキテクチャの制約（約15\%の寄与）の3つの要因が複合的に作用していると結論づけられる。特に、肘の体系的バイアスやカメラ視点依存性は、学習データの偏りに直接起因するものであり、より多様な姿勢やカメラ視点を含むデータセットで追加学習（ファインチューニング）を行うことで改善が期待できる。一方、腰の左右対称な誤差や足首・肩の高変動誤差は、単眼カメラの原理的限界や2Dキーポイント検出の不安定性に起因するため、モデルアーキテクチャの根本的な改良や、複数カメラの統合が必要である。

本研究の大規模データセット（259,356観測）を用いた定量的分析により、MediaPipeの推定誤差が単なるランダムノイズではなく、体系的なパターンを持つことが明らかになった。これらの知見は、姿勢推定の精度向上のための具体的な方向性を示すものであり、学習データの設計、モデルアーキテクチャの改良、後処理による補正など、複数のアプローチを組み合わせた総合的な改善戦略が必要であることを示唆している。

\begin{figure}[h]
\centering
\fbox{\parbox{0.4\textwidth}{\centering [図6: 推定精度が低下した例]}}
\caption{推定精度が低下した例}
\end{figure}

\section{まとめ}

本研究では、Unity環境で取得したグラウンドトゥルース（GT）データを用いて、MediaPipeの3次元姿勢推定精度を定量評価した。505台のマルチビューカメラから107フレームの歩行データを収集し、12関節について合計259,356観測を得た。評価指標として、関節角度誤差（Joint Angle MAE）と方向角誤差（Direction Detection Error, $\Delta\theta$・$\Delta\psi$）の二種類を採用し、表・図で区別して報告した。

関節角度誤差（MAE）では、8関節（左右肩・肘・腰・膝）について平均絶対誤差を算出した。肩で約34°、肘・膝で約14°、腰で約20°となり、肘と膝は相対的に誤差が小さく、肩は大きい傾向を示した。一方、方向角誤差に基づく相関分析では、肘関節で$\Delta\theta$が左右対称に約$\pm121°$の体系的バイアスを持つこと、腰関節で奥行き方向（$\Delta\psi$）の誤差が強い負の相関（r = -0.84）を示すこと、上肢の肩-肘-手首が同方向に連動して誤差を持つことが明らかになった。これらはMediaPipeの誤差がランダムではなく、学習データの偏り・単眼カメラの奥行き推定限界・階層的推定に起因する体系的なパターンを持つことを示している。また、同一姿勢でもカメラ視点により精度が大きく変動し、視点依存性が定量的に確認された。

以上の知見は、実用場面でカメラ配置や視点が限定できない場合に、MediaPipeの適用範囲と限界を把握するうえで有用である。今後の課題として、カメラアングルや照明条件の変化、複数の動作パターンなど、より多様な条件下での評価を拡張すること、および肘・腰の体系的誤差を補正する後処理や、結果に基づくライブラリ・ツールの開発を検討する。

\begin{thebibliography}{9}
\bibitem{mediapipe}
Google, "MediaPipe Pose," \textit{https://ai.google.dev/edge/mediapipe/solutions/vision/pose\_landmarker}

\bibitem{blazepose}
V. Bazarevsky et al., "BlazePose: On-device Real-time Body Pose tracking," \textit{arXiv preprint arXiv:2006.10204}, 2020.

\bibitem{unity}
Unity Technologies, "HumanBodyBones," \textit{Unity Documentation}, 2024.
\end{thebibliography}

\end{document}