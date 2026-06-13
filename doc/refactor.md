# Refine段階 高速化計画

## 現状の課題

GeneratePoseで描画したマスクを使ったNelder-Meadによる6DOF pose推定において、
Refine段階がボトルネックとなっている。

### コスト評価の内訳（推定、NVIDIA GPU）

```
GPU Render (~0.5-2ms)
  → 同期待ち
  → RGBA Readback 1.6MB (~0.5-2ms)
  → RgbaToMask CPU (~0.3-1ms)
  → cv::distanceTransform (~1-3ms)
  → absdiff + max + mul + sum (~0.5-1ms)
  → countNonZero (~0.2ms)
= 約3-10ms/eval × 200-500eval (Nelder-Mead 6D) = 0.6-5秒/候補
```

### 却下された案と理由

| 案 | 理由 |
|----|------|
| マルチスレッド有効化 | GeneratePose自体が高コストで根本解決にならない |
| LM/GNデフォルト化 | 過去試したがスコアが大きく低下した |
| Generate→GeneratePose修正 | benchmark差がほぼない |

---

## 実装プラン（4段階、インパクト順）

### Step 1: R8単一チャンネルレンダリング

**対象**: `mask-generation/src/vulkan_context.cpp` のレンダーパイプライン

**変更内容**:

- カラーアタッチメント: `VK_FORMAT_R8G8B8A8_UNORM` → `VK_FORMAT_R8_UNORM`
- Fragment shader: 白黒直接出力（0 or 255）
- リードバックバッファサイズ: `width * height * 4` → `width * height`
- `RgbaToMask()` 削除。readbackデータをそのまま `cv::Mat(CV_8UC1)` にコピー

**効果**:
- リードバック量 1/4（1.6MB → 400KB）
- RgbaToMask CPUループ削除
- ~1-2ms/eval 削減

**作業規模**: 半日、リスク低

---

### Step 2: BOBYQAへの最適化手法変更

**対象**: `src/cached_pose_estimator.cpp` の RefinePose、新規 `src/bobyqa.h`

**変更内容**:

- `nelder_mead.h` と同様にヘッダオンリーで BOBYQA (Bound Optimization BY Quadratic Approximation) を実装
  - 代替: NLopt（`nlopt::algorithm::LN_BOBYQA`）を依存関係に追加
- 6D問題での評価回数: Nelder-Mead ~200-500 → BOBYQA ~50-100
- コスト関数はそのまま流用（微分不要）

**効果**:
- 評価回数 1/3-1/5 に削減
- 収束品質はNelder-Meadと同等が期待される（二次モデル近似による超線形収束）

**精度検証方法**:
- 同じ初期候補に対して Nelder-Mead と BOBYQA の最終IoUを比較
- スコアが同等以上であることを確認してからデフォルト化

**作業規模**: 1-2日（精度検証含む）

---

### Step 3: GPU Compute Shader によるコスト計算

**対象**: `mask-generation/src/vulkan_context.cpp`、新規コンピュートシェーダー

**変更内容**:

Vulkan Compute パイプラインを追加し、コスト計算全体をGPUで実行:

```
[入力]
  - GPU上のレンダリング結果（R8イメージ）
  - 入力マスクのDT（Estimate()開始時に一度だけGPUへアップロード）

[Compute Shader 1] 二値化 + absdiff
  - レンダリング結果と入力マスクのピクセル単位差分

[Compute Shader 2] Distance Transform
  - Two-pass sweepアルゴリズム（水平 + 垂直）
  - レンダリングマスクのDTをGPU上で計算

[Compute Shader 3] Chamfer cost
  - max(dt_input, dt_rendered) * diff → parallel sum reduction

[Compute Shader 4] Area counting
  - countNonZero → parallel sum reduction

[出力]
  - スカラー値2つ: chamfer_val, area
  - リードバック: 8 bytesのみ
```

**効果**:
- リードバック量: 1.6MB → 8 bytes（200,000分の1）
- CPU側処理を完全削除: RgbaToMask, absdiff, distanceTransform, chamfer, countNonZero
- ~3-10ms/eval 削減

**技術的課題**:
- GPU上のDistance Transform実装が最も複雑
- Two-pass sweep: 行方向→列方向の2パスでO(n)計算
- 別案: Jump Flooding Algorithm (JFA)、O(log n)パス

**作業規模**: 3-5日

---

### Step 4: 低解像度での初期最適化

**対象**: `src/cached_pose_estimator.cpp` の RefinePose

**変更内容**:

- 解像度 1/4（例: 848x480 → 212x120）の `MaskGenerator` を追加作成
- 入力マスクも 1/4 に縮小してDT事前計算
- 最適化前半（収束前）は低解像度でコスト評価
- IoUが閾値を超えたらフル解像度に切替

**効果**:
- 初期イテレーション ~4x高速化（画素数 1/4）
- chamfer距離はスケールに概ね不変のため精度への影響は小さい

**作業規模**: 半日（Step 1-3完了後）

---

## 組み合わせ効果の推定

| 構成 | 評価回数 | evalあたり時間 | Refine合計(1候補) |
|------|---------|-------------|-------------------|
| 現状 | ~300 | ~5ms | ~1500ms |
| Step 1+2 | ~80 | ~3ms | ~240ms |
| Step 1+2+3 | ~80 | ~0.5-1ms | ~40-80ms |
| Step 1+2+3+4 | ~80 | ~0.3-0.8ms (avg) | ~25-60ms |

---

## 実装順序

1. **Step 1** (R8) → リスク低、即効性
2. **Step 2** (BOBYQA) → 精度検証が必要だが評価回数削減効果大
3. **Step 3** (GPU Compute) → 最大インパクトだがDT実装が複雑
4. **Step 4** (低解像度) → Step 1-3完了後に追加

各Step完了時に既存のテスト・ベンチマークで回帰がないことを確認する。
