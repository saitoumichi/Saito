# 高解像度胸部CT画像の非剛体レジストレーション  
## 3次元Haarウェーブレット変換・完全再構成・VoxelMorphを用いた検証

## 1. 研究概要

本研究では、胸部CT画像の経時比較を支援するために、**3次元ウェーブレット変換と深層学習ベースの非剛体画像レジストレーションを組み合わせた手法**を検討する。

対象とするのは、呼吸状態や体位の違いによって形状が変化した2時点の胸部CT画像である。基準画像を **Fixed image**、位置合わせの対象画像を **Moving image** とし、Moving imageをFixed imageへ変形するための変位ベクトル場（Displacement Vector Field: DVF）を推定する。

本研究では、特に次の点を重点的に検証する。

- 3次元Haarウェーブレット変換による周波数分解
- Analysisフィルタ、ダウンサンプリング、Synthesisフィルタの処理順
- ウェーブレット変換の完全再構成性
- 256×256×256画素の高解像度CT画像への適用
- VoxelMorphを用いたDVF推定
- 周波数成分を利用した画像レジストレーション
- カリキュラム学習による変形量の段階的増加
- RMSE、MS-SSIM、Dice、folding rateによる定量評価

---

## 2. 研究背景

胸部CT画像は、肺がん、間質性肺炎、間質性肺異常などの呼吸器疾患において、診断、治療計画、病態進行の確認に広く用いられている。

臨床現場では、現在のCT画像と過去のCT画像を比較し、病変の進行、退縮、形態変化を評価する。しかし、異なる撮影時点では、次のような要因によって画像間に位置ずれが生じる。

- 吸気量や呼気量の違い
- 患者の体位の違い
- 肺や胸郭の非線形な変形
- 撮影範囲やスライス位置の違い
- 臓器や血管の局所的な移動

そのため、画像を単純に重ねるだけでは、同じ解剖学的位置を正確に比較できない。

この問題を解決するために、Moving imageをFixed imageへ変形する**非剛体画像レジストレーション**を行う。

---

## 3. 研究目的

本研究の目的は、既存のウェーブレット変換を用いた胸部CT画像レジストレーション手法について、**ウェーブレット分解と再構成の処理順を整理し、完全再構成可能な構成へ修正したうえで、その有効性を定量的に評価すること**である。

特に、従来実装では次のような処理順が含まれていた。

```text
入力画像
→ 通常のダウンサンプリング
→ 周波数分解
→ 周波数成分の加算
→ アップサンプリング
```

しかし、離散ウェーブレット変換の基本構造では、次の順序で処理する必要がある。

```text
入力画像
→ Analysisフィルタによる周波数分解
→ ダウンサンプリング
→ ネットワーク処理
→ アップサンプリング
→ Synthesisフィルタによる再構成
```

本研究では、この処理順を実装へ反映し、以下を検証する。

1. 3次元Haarウェーブレット変換による正しい周波数分解
2. 逆ウェーブレット変換による完全再構成
3. 分解前後における数値誤差
4. 周波数成分を利用したDVF推定
5. 高解像度CT画像への拡張
6. レジストレーション精度と変形場の妥当性

---

## 4. 画像レジストレーション

### 4.1 Fixed imageとMoving image

本研究では、2枚のCT画像を入力する。

- **Fixed image**  
  基準となる画像。Moving imageをこの画像へ合わせる。

- **Moving image**  
  変形対象となる画像。推定したDVFによって変形する。

ネットワークは、Fixed imageとMoving imageの組からDVFを推定する。

```text
Moving image + Fixed image
            ↓
         Network
            ↓
            DVF
```

推定したDVFをSpatial Transformerへ入力し、Moving imageを変形する。

```text
Moving image + DVF
          ↓
Spatial Transformer
          ↓
      Moved image
```

Moved imageがFixed imageに近づくようにネットワークを学習する。

---

## 5. VoxelMorph

本研究では、深層学習ベースの画像レジストレーションモデルとしてVoxelMorphを利用する。

VoxelMorphは、Moving imageとFixed imageの組を入力し、両画像を対応付けるDVFを推定するネットワークである。

一般的な反復最適化型レジストレーションでは、画像ペアごとに最適化処理を行う。一方、VoxelMorphでは、学習データ全体を用いてネットワークパラメータを学習する。そのため、学習後は新しい画像ペアに対して1回の推論でDVFを出力できる。

本研究における基本的な流れは次のとおりである。

```text
Moving image
Fixed image
    ↓
U-Net型ネットワーク
    ↓
3チャネルDVF
    ↓
Spatial Transformer
    ↓
Moved image
```

3次元画像では、DVFは各ボクセルに対して次の3方向の変位を持つ。

```text
dx : 幅方向の変位
dy : 高さ方向の変位
dz : 奥行き方向の変位
```

したがって、DVFのチャネル数は3である。

---

## 6. 3次元ウェーブレット変換

### 6.1 ウェーブレット変換を使用する理由

胸部CT画像には、大きな構造と細かい構造が同時に含まれる。

- 肺全体や胸郭の形状：低周波情報
- 血管、気管支、臓器境界：高周波情報
- ノイズや細かな濃度変化：高周波情報

ウェーブレット変換を用いることで、画像を異なる周波数成分へ分解できる。

本研究では、3次元CT画像に対して、3次元Haarウェーブレット変換を適用する。

---

### 6.2 Haar Analysisフィルタ

1次元Haarウェーブレットの低域Analysisフィルタと高域Analysisフィルタは、次式で表される。

\[
h_L = \frac{1}{\sqrt{2}}[1, 1]
\]

\[
h_H = \frac{1}{\sqrt{2}}[1, -1]
\]

低域フィルタは隣接値の和に相当し、信号の大まかな変化を取り出す。

高域フィルタは隣接値の差に相当し、信号の細かな変化や境界を取り出す。

---

### 6.3 3次元への拡張

3次元画像には、奥行き、高さ、幅の3方向がある。

各方向に低域フィルタ \(L\) または高域フィルタ \(H\) を適用するため、次の8種類のサブバンドが得られる。

```text
LLL
LLH
LHL
LHH
HLL
HLH
HHL
HHH
```

それぞれの意味は、3軸に適用した低域・高域フィルタの組合せである。

例：

\[
K_{LLL} = h_L \otimes h_L \otimes h_L
\]

\[
K_{LLH} = h_L \otimes h_L \otimes h_H
\]

\[
K_{HHH} = h_H \otimes h_H \otimes h_H
\]

ここで、\(\otimes\) は各軸方向のフィルタを組み合わせることを表す。

Haarフィルタは1方向につき2タップであるため、3次元Analysisフィルタの実効的なサイズは、

\[
2 \times 2 \times 2
\]

となる。

---

### 6.4 サブバンドの役割

| サブバンド | 主な情報 |
|---|---|
| LLL | 全方向の低周波情報。臓器全体や肺の大まかな形状 |
| LLH | 1方向の高周波情報 |
| LHL | 1方向の高周波情報 |
| HLL | 1方向の高周波情報 |
| LHH | 2方向の高周波情報 |
| HLH | 2方向の高周波情報 |
| HHL | 2方向の高周波情報 |
| HHH | 全方向の高周波情報。角、細かな構造、急激な濃度変化 |

各文字がどの軸に対応するかは実装の軸順序によって異なるため、コード上の次元順を確認する必要がある。

---

### 6.5 ダウンサンプリング

ウェーブレット変換では、Analysisフィルタによる畳み込み後、各軸で2個に1個のサンプルを残す。

入力が、

\[
256 \times 256 \times 256
\]

の場合、各サブバンドは、

\[
128 \times 128 \times 128
\]

となる。

ただし、8個のサブバンドが得られるため、全係数数は元画像と等しい。

\[
8 \times 128^3 = 256^3
\]

したがって、単純な画像縮小とは異なり、元画像の情報を8個の周波数成分へ再配置している。

---

## 7. Analysis処理

Analysis処理では、入力画像を低域・高域フィルタで畳み込み、その後ダウンサンプリングする。

1次元の場合は、

\[
A[n]
=
\sum_k \tilde{h}(2n-k)x[k]
\]

\[
D[n]
=
\sum_k \tilde{g}(2n-k)x[k]
\]

と表せる。

ここで、

- \(A[n]\)：低周波の近似成分
- \(D[n]\)：高周波の詳細成分
- \(\tilde{h}\)：低域Analysisフィルタ
- \(\tilde{g}\)：高域Analysisフィルタ

である。

処理の流れは次のとおり。

```text
入力信号
  ├─ 低域フィルタ → 2個に1個残す → 近似成分
  └─ 高域フィルタ → 2個に1個残す → 詳細成分
```

3次元では、この処理を3軸へ分離可能に適用し、8サブバンドを生成する。

---

## 8. Synthesis処理と逆ウェーブレット変換

逆ウェーブレット変換では、各サブバンドをアップサンプリングし、Synthesisフィルタを適用して加算する。

1次元の場合は、

\[
x[n]
=
\sum_k h(n-2k)A[k]
+
\sum_k g(n-2k)D[k]
\]

と表せる。

処理の流れは次のとおり。

```text
低周波成分 → 0挿入 → 低域Synthesisフィルタ ┐
                                                  ├→ 加算 → 再構成信号
高周波成分 → 0挿入 → 高域Synthesisフィルタ ┘
```

Haar変換では、

\[
a_n=\frac{x_{2n}+x_{2n+1}}{\sqrt{2}}
\]

\[
d_n=\frac{x_{2n}-x_{2n+1}}{\sqrt{2}}
\]

と分解した場合、逆変換は、

\[
x_{2n}=\frac{a_n+d_n}{\sqrt{2}}
\]

\[
x_{2n+1}=\frac{a_n-d_n}{\sqrt{2}}
\]

となる。

---

## 9. 完全再構成

完全再構成とは、ウェーブレット変換によって分解した信号を逆変換したとき、元の信号を誤差なく復元できる性質である。

理想的には、

\[
\hat{x}=x
\]

である。

数値計算では浮動小数点誤差が生じるため、次の誤差を計算する。

\[
e_{\mathrm{mean}}
=
\frac{1}{N}
\sum_i
|x_i-\hat{x}_i|
\]

\[
e_{\mathrm{max}}
=
\max_i
|x_i-\hat{x}_i|
\]

これまでの確認では、以下のような非常に小さい誤差が得られている。

### 128×256×256データ

```text
平均誤差：約 1.37 × 10^-8
最大誤差：約 2.38 × 10^-7
```

### 256×256×256データ

```text
平均誤差：約 5.37 × 10^-8
最大誤差：約 9.54 × 10^-7
```

別の検証例では、

```text
mean error: 1.460843357392605e-08
max error : 3.5762786865234375e-07
```

が得られている。

これらはfloat32演算に伴う丸め誤差の範囲であり、実装上はほぼ完全再構成できていると判断できる。

---

## 10. 完全再構成条件

2チャネルフィルタバンクでは、完全再構成のために主に次の点を確認する。

### 10.1 エイリアシングキャンセル

ダウンサンプリングによって発生する折り返し成分が、Synthesis処理で打ち消される必要がある。

一般的には、

\[
G_0(z)H_0(-z)+G_1(z)H_1(-z)=0
\]

のような条件で表される。

---

### 10.2 振幅歪みの抑制

低域経路と高域経路を合成したとき、元信号の振幅特性を保つ必要がある。

\[
G_0(z)H_0(z)+G_1(z)H_1(z)
=
c z^{-k}
\]

ここで、

- \(c\)：一定の倍率
- \(z^{-k}\)：一定の遅延

である。

---

### 10.3 AnalysisフィルタとSynthesisフィルタの対応

直交Haarウェーブレットでは、Analysis側とSynthesis側のフィルタが適切な反転・符号関係を持つ。

実装では、フィルタ係数、軸順、符号、畳み込みと相関の違いを確認する必要がある。

---

## 11. 提案手法の全体構成

本研究で対象とする処理の全体像は次のとおり。

```text
Moving image : 256×256×256
Fixed image  : 256×256×256
             ↓
3D Haar Wavelet Transform
             ↓
Moving : 8サブバンド、各128×128×128
Fixed  : 8サブバンド、各128×128×128
             ↓
Generator / VoxelMorph
             ↓
DVF : 3×128×128×128
             ↓
Spatial Transformer
             ↓
変形後Moving側サブバンド
             ↓
Inverse 3D Haar Wavelet Transform
             ↓
Moved image : 256×256×256
```

ネットワークは、Moving側とFixed側の周波数成分からDVFを推定する。

その後、Moving側サブバンドをSpatial Transformerで変形し、逆ウェーブレット変換によって高解像度のMoved imageを再構成する。

---

## 12. DVFとSpatial Transformer

DVFは、各ボクセルをどの方向へどれだけ移動するかを表す。

\[
\mathbf{u}(x,y,z)
=
\left(
u_x(x,y,z),
u_y(x,y,z),
u_z(x,y,z)
\right)
\]

Moving imageを \(M\)、変形後画像を \(M'\) とすると、

\[
M'(x,y,z)
=
M
\left(
x+u_x,
y+u_y,
z+u_z
\right)
\]

のように表される。

移動先が整数ボクセル位置になるとは限らないため、Spatial Transformerでは三線形補間を使用する。

---

## 13. カリキュラム学習

本研究のベース手法では、カリキュラム学習を使用する。

初めから大きく複雑な変形を学習させるのではなく、最初は小さく簡単な変形から学習し、徐々に変形量を大きくする。

粗いDVFは、

\[
8\times8\times8
\]

の制御点グリッド上で生成する。

各制御点の変位は、一様分布

\[
U(-A,A)
\]

からサンプリングする。

生成した粗いDVFを三線形補間により、

\[
128\times128\times128
\]

へアップサンプリングする。

入力サイズが128×128×128の場合、制御点間隔は16ボクセルに相当する。

カリキュラム学習では、変形の最大振幅 \(A\) を段階的に増やす。

```text
小さい変形
→ 中程度の変形
→ 大きい変形
```

これにより、ネットワークが急に難しい変形へ適応できなくなることを防ぐ。

---

## 14. 損失関数

### 14.1 画像類似度損失

Moved imageとFixed imageが近くなるようにする。

例：

\[
L_{\mathrm{sim}}
=
\frac{1}{N}
\sum_i
\left(
F_i-M'_i
\right)^2
\]

または、局所正規化相互相関（LNCC）を用いる。

---

### 14.2 変形場平滑化損失

DVFが急激に変化しないようにする。

\[
L_{\mathrm{def}}
=
\sum_x
\|\nabla \mathbf{u}(x)\|^2
\]

---

### 14.3 総損失

ベース手法では、次のような重み付き和を用いる。

\[
L
=
\alpha L_{\mathrm{sim}}
+
\beta L_{\mathrm{def}}
\]

例：

```text
α = 100
β = 0.01
```

Fine-tuning時には、設定によって \(\beta=0\) とする。

---

### 14.4 逆写像整合性

今後、Moving→FixedとFixed→Movingの変形が互いに逆になるよう制約する逆写像整合性の導入を検討する。

\[
\phi_{MF}\circ\phi_{FM}
\approx I
\]

ここで \(I\) は恒等変換である。

---

## 15. データセット

現在使用しているデータの例：

```text
症例数：501
訓練用：400
評価用：101
```

胸部CT画像は、元データとして512×512画素を持つ。

確認済みデータ例：

```python
x_train.shape == (400, 128, 256, 256)
x_train.dtype == float64
```

ファイル例：

```text
TrainData_NoBed.npz
CT_Train_512_FOV
CT_Train_MI_NoBed
```

実際のファイル名や保存場所は環境ごとに異なる。

---

## 16. 評価指標

### 16.1 RMSE

画像の画素値差を評価する。

\[
\mathrm{RMSE}
=
\sqrt{
\frac{1}{N}
\sum_i
(F_i-M'_i)^2
}
\]

CT画像ではHU単位で評価する。

小さいほどよい。

---

### 16.2 MS-SSIM

複数スケールにおける画像の構造類似度を評価する。

1に近いほどよい。

---

### 16.3 Dice係数

肺領域などのセグメンテーションマスクの重なりを評価する。

\[
\mathrm{Dice}
=
\frac{2|A\cap B|}{|A|+|B|}
\]

1に近いほどよい。

---

### 16.4 Folding rate

DVFによって局所的な折り返しが発生していないかを評価する。

変形写像のJacobian determinantが負になる領域は、解剖学的に不自然な折り返しを示す。

\[
\det(J_\phi)<0
\]

となるボクセルの割合をfolding rateとして評価する。

小さいほどよい。

---

## 17. 現在の評価結果

1ペアでの評価例：

```text
pair   : pair101
RMSE   : 151.218094
MS-SSIM: 0.967091
Dice   : 0.960459
```

ベース手法で報告されている例：

```text
RMSE        : 126.37 ± 82.03 HU
MS-SSIM     : 0.990
Dice        : 0.974
Folding rate: 9.12 %
```

現時点の自己実験は1ペアのみの結果であるため、今後は評価用101ペア全体で統計的に評価する必要がある。

---

## 18. 実装環境

確認済み環境例：

```text
OS       : Windows / macOS
Python   : 3.10
Framework: PyTorch
GPU      : NVIDIA GeForce RTX 3060 Ti
Backend  : VXM_BACKEND=pytorch
```

主要ライブラリ：

```text
torch
numpy
scipy
matplotlib
pandas
tqdm
voxelmorph
pytorch-msssim
```

---

## 19. モデル設定例

```python
nb_features = [
    [32, 64, 64, 64, 64],
    [64, 64, 64, 64, 64, 32, 16, 16]
]
```

モデル例：

```python
model = VxmDense_128_256_256(
    inshape=(128, 256, 256),
    nb_unet_features=nb_features,
    int_steps=0
)
```

Spatial Transformer例：

```python
transformer = SpatialTransformer(
    size=(128, 256, 256)
)
```

実際のクラス名、引数、入力サイズは使用中の実装に合わせて変更する必要がある。

---

## 20. リポジトリ構成例

```text
.
├── README.md
├── data/
│   ├── train/
│   └── test/
├── models/
│   ├── voxelmorph.py
│   ├── wavelet_model.py
│   └── spatial_transformer.py
├── wavelet/
│   ├── haar3d.py
│   ├── analysis.py
│   ├── synthesis.py
│   └── reconstruction_test.py
├── train/
│   ├── train_pretrain.py
│   ├── train_finetune.py
│   └── curriculum.py
├── evaluate/
│   ├── evaluate_pairs.py
│   ├── metrics.py
│   └── visualize.py
├── notebooks/
│   └── experiments.ipynb
├── checkpoints/
└── results/
```

この構成は説明用の例であり、現在のリポジトリ構成に合わせて修正する。

---

## 21. 実行手順例

### 21.1 環境構築

```bash
conda create -n vxm310 python=3.10
conda activate vxm310
pip install torch numpy scipy matplotlib pandas tqdm pytorch-msssim
```

VoxelMorphをGitHub版から使用する場合は、対象リポジトリの手順に従う。

---

### 21.2 完全再構成テスト

```bash
python wavelet/reconstruction_test.py
```

確認項目：

```text
input shape
subband shape
reconstructed shape
mean error
max error
```

期待される形状：

```text
Input       : 256×256×256
Each band   : 128×128×128
Bands       : 8
Reconstructed: 256×256×256
```

---

### 21.3 学習

```bash
python train/train_pretrain.py
```

Fine-tuning：

```bash
python train/train_finetune.py
```

---

### 21.4 評価

```bash
python evaluate/evaluate_pairs.py
```

出力例：

```text
pair
RMSE
MS-SSIM
Dice
folding rate
```

---

## 22. 現在までの進捗

- 3次元Haarウェーブレット変換を実装
- 8サブバンドへの分解を確認
- 逆ウェーブレット変換を実装
- 128×256×256および256×256×256で再構成誤差を確認
- float32範囲でほぼ完全再構成できることを確認
- pair101に対するRMSE、MS-SSIM、Diceを計算
- 複数ペア評価用コードを作成中
- Analysis→Downsampling→Upsampling→Synthesisの処理順へ修正中
- 256画素対応DVFおよびモデル構造を検討中
- SWTとDWTの比較を検討中
- 逆写像整合性制約の導入を検討中

---

## 23. 今後の課題

1. 学習コードへ正しいDWT処理順を反映する
2. AnalysisフィルタとSynthesisフィルタの対応を確認する
3. 完全再構成条件を数式とコードの両面から確認する
4. 101評価ペア全体でRMSE、MS-SSIM、Diceを計算する
5. Jacobian determinantとfolding rateを評価する
6. 256×256×256に対応したDVF推定を実現する
7. DWTとSWTの再構成誤差・計算量・精度を比較する
8. 1段階分解と2段階分解を比較する
9. 逆写像整合性制約を導入する
10. GPUメモリ使用量と学習時間を評価する
11. 高周波成分がレジストレーション精度へ与える影響を分析する
12. 各サブバンドを除外したアブレーション実験を行う

---

## 24. 研究上の重要な整理

本研究で用いるウェーブレット変換は、単なる画像縮小ではない。

```text
通常のダウンサンプリング
→ 一部の情報を捨ててサイズを小さくする
```

```text
離散ウェーブレット変換
→ 低周波・高周波へ情報を分ける
→ 各サブバンドをダウンサンプリングする
→ 全サブバンドを保持すれば元画像を再構成できる
```

また、現在の処理は1段階の3次元ウェーブレット分解である。

明確な多段階分解を行う場合は、LLL成分をさらに分解する。

```text
256³
↓ 1段階目
LLL 128³ + 高周波7成分
↓ LLLを再分解
LLL 64³ + 高周波7成分
```

---

## 25. 参考文献

1. S. G. Mallat,  
   “A Theory for Multiresolution Signal Decomposition: The Wavelet Representation,”  
   *IEEE Transactions on Pattern Analysis and Machine Intelligence*,  
   vol. 11, no. 7, pp. 674–693, 1989.

2. G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, and A. V. Dalca,  
   “VoxelMorph: A Learning Framework for Deformable Medical Image Registration,”  
   *IEEE Transactions on Medical Imaging*, vol. 38, no. 8, pp. 1788–1800, 2019.

3. I. Daubechies,  
   “Orthonormal Bases of Compactly Supported Wavelets,”  
   *Communications on Pure and Applied Mathematics*, vol. 41, no. 7, pp. 909–996, 1988.

4. Haar wavelet、完全再構成フィルタバンク、医用画像レジストレーションに関する関連文献を今後追加する。

---

## 26. 注意事項

- 医用画像データは個人情報を含む可能性があるため、公開リポジトリへアップロードしない。
- 学習済みモデルや大容量NPZファイルはGit管理対象から除外する。
- `.gitignore`へデータセット、チェックポイント、ログを追加する。
- 軸順序 `(D, H, W)` と `(H, W, D)` の混同に注意する。
- PyTorchの`Conv3d`は相関演算として実装されているため、理論上の畳み込みとのフィルタ反転の違いを確認する。
- Analysis側とSynthesis側で符号や並び順が一致しているかを確認する。
- サブバンド名とテンソルの格納順を明示する。

---

## 27. まとめ

本研究では、胸部CT画像の高精度な非剛体レジストレーションを目的として、3次元Haarウェーブレット変換とVoxelMorphを組み合わせる。

ウェーブレット変換によって、256×256×256のCT画像を8個の128×128×128サブバンドへ分解する。ネットワークは周波数分解されたMoving imageとFixed imageからDVFを推定し、Moving側サブバンドを変形する。最後に逆ウェーブレット変換を行い、256×256×256のMoved imageを再構成する。

本研究の中心課題は、ウェーブレット変換の処理順と完全再構成性を正しく実装し、そのうえで高解像度CT画像レジストレーションの精度、構造類似度、領域重なり、変形場の妥当性を定量的に評価することである。
