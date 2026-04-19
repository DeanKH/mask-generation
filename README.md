# maskgen

Vulkan を用いたオフスクリーンレンダリングによる、3Dメッシュのマスク画像生成ライブラリおよびCLIツール。

カメラの内部パラメータ・外部パラメータと三角形メッシュ（PLY / OBJ）を入力とし、描画領域を白・背景を黒とした2値マスク画像を `cv::Mat` 型で出力する。

## 動作環境

| 項目 | 内容 |
|------|------|
| OS | Ubuntu 24.04（Docker） |
| GPU | Intel iGPU（Broadwell以降）または任意のVulkan対応GPU |
| Vulkanドライバ | Mesa ANV（Intel）/ その他Vulkan準拠ドライバ |
| Docker | `/dev/dri` デバイスの渡しが必要 |

## クイックスタート

```bash
# ビルド
docker compose build

# デフォルト設定でマスク画像生成
docker compose run --rm maskgen /workspace/model/bun_zipper_res4.ply -o /workspace/output.png

# カメラパラメータを指定して生成
docker compose run --rm maskgen /workspace/model/bun_zipper_res4.ply \
  --eye 0 0.13 0.3 \
  --target 0 0.13 0 \
  --fx 800 --fy 800 \
  -o /workspace/output.png
```

出力ファイルは `/workspace/` にホストのプロジェクトルートがマウントされているため、ホスト側から直接参照可能。

## CLI オプション

```
Usage: maskgen_cli [OPTIONS] <mesh_file>

Options:
  -o, --output PATH      出力PNGパス (default: mask.png)
  --width INT            画像幅 (default: 640)
  --height INT           画像高さ (default: 480)
  --fx FLOAT             焦点距離x (default: 500.0)
  --fy FLOAT             焦点距離y (default: 500.0)
  --cx FLOAT             主点x (default: width/2)
  --cy FLOAT             主点y (default: height/2)
  --near FLOAT           近クリップ面 (default: 0.01)
  --far FLOAT            遠クリップ面 (default: 100.0)
  --eye X Y Z            カメラ位置 (default: 0 0 1)
  --target X Y Z         注視点 (default: 0 0 0)
  --up X Y Z             上方向ベクトル (default: 0 1 0)
  --mesh-tx FLOAT        メッシュ平行移動x (default: 0)
  --mesh-ty FLOAT        メッシュ平行移動y (default: 0)
  --mesh-tz FLOAT        メッシュ平行移動z (default: 0)
  --mesh-rx FLOAT        メッシュ回転x [degree] (default: 0)
  --mesh-ry FLOAT        メッシュ回転y [degree] (default: 0)
  --mesh-rz FLOAT        メッシュ回転z [degree] (default: 0)
  -h, --help             ヘルプを表示
```

## プロジェクト構成

```
mask-generation/
├── CMakeLists.txt                 ルートCMake（install対応）
├── Dockerfile                     Ubuntu 24.04 + Vulkan + OpenCV + glm + shaderc
├── docker-compose.yml
├── .clang-format                  Google C++ Style
├── .clang-tidy
├── cmake/
│   └── maskgenConfig.cmake.in     外部CMakeプロジェクト用パッケージ設定
├── shaders/
│   ├── mask.vert                  頂点シェーダ（MVP変換）
│   └── mask.frag                  フラグメントシェーダ（白色出力）
├── include/maskgen/
│   ├── camera.h                   CameraParams 構造体
│   ├── mesh.h                     Mesh クラス（PLY/OBJ読み込み）
│   └── mask_generator.h           MaskGenerator クラス（Pimpl）
├── src/
│   ├── CMakeLists.txt             libmaskgen.so（SHAREDライブラリ）
│   ├── vulkan_context.h/cpp       Vulkanオフスクリーンレンダリング
│   ├── mask_generator.cpp         公開API実装（MVP行列計算）
│   └── mesh.cpp                   PLY / OBJパーサー
├── app/
│   ├── CMakeLists.txt             CLI実行ファイル
│   └── main.cpp                   コマンドラインプログラム
└── model/
    └── bun_zipper_res4.ply        サンプルメッシュ
```

## ライブラリ API

### CameraParams

```cpp
#include <maskgen/camera.h>

maskgen::CameraParams params;
params.width = 640;
params.height = 480;
params.fx = 500.0;
params.fy = 500.0;
params.cx = 320.0;
params.cy = 240.0;
params.near_plane = 0.01;
params.far_plane = 100.0;
params.eye_x = 0.0;
params.eye_y = 0.0;
params.eye_z = 1.0;
params.target_x = 0.0;
params.target_y = 0.0;
params.target_z = 0.0;
params.up_x = 0.0;
params.up_y = 1.0;
params.up_z = 0.0;
```

### MeshPose

```cpp
#include <maskgen/mask_generator.h>

maskgen::MeshPose pose;
pose.tx = 0.0;   // 平行移動 x
pose.ty = 0.0;   // 平行移動 y
pose.tz = 0.0;   // 平行移動 z
pose.rx = 0.0;   // 回転 x [rad]
pose.ry = 0.0;   // 回転 y [rad]
pose.rz = 0.0;   // 回転 z [rad]
```

### 使用例

```cpp
#include <maskgen/camera.h>
#include <maskgen/mask_generator.h>
#include <maskgen/mesh.h>

#include <opencv2/imgcodecs.hpp>

int main() {
  maskgen::CameraParams params;
  params.fx = 800.0;
  params.fy = 800.0;
  params.eye_y = 0.13;
  params.eye_z = 0.3;
  params.target_y = 0.13;

  maskgen::Mesh mesh;
  mesh.LoadFromFile("model.ply");

  maskgen::MeshPose pose;

  maskgen::MaskGenerator generator(params);
  cv::Mat mask = generator.Generate(mesh, pose);

  cv::imwrite("mask.png", mask);
  return 0;
}
```

## 外部CMakeプロジェクトからの利用

`cmake --install` でインストール後、別プロジェクトから `find_package` で利用可能。

```cmake
find_package(maskgen REQUIRED)
target_link_libraries(my_app PRIVATE maskgen)
```

## 依存ライブラリ

| ライブラリ | 用途 | ライセンス |
|-----------|------|-----------|
| Vulkan | オフスクリーンレンダリング | Apache 2.0 |
| OpenCV | 画像データ表現（cv::Mat）・PNG出力 | Apache 2.0 |
| GLM | 行列・ベクトル演算 | MIT |
| shaderc | GLSLからSPIR-Vへのランタイムコンパイル | Apache 2.0 |
| Mesa (ANV) | Intel iGPU向けVulkanドライバ | MIT |

## 設計

### Vulkanオフスクリーンレンダリング

- ヘッドレス環境対応。ウィンドウシステムに依存しない
- カラーアタッチメント（R8G8B8A8）+ デプスアタッチメント（D32）のフレームバッファ
- レンダリング後にホスト可視バッファへコピーして `cv::Mat` に変換
- シェーダはランタイムでコンパイル（shaderc利用）し、プリコンパイル済みSPIR-Vの配布が不要

### 投影行列

カメラの内部パラメータ（fx, fy, cx, cy）から OpenGL/Vulkan 互換の射影行列を直接構成する。

```
| 2*fx/w    0           1-2*cx/w    0          |
| 0         -2*fy/h     1-2*cy/h    0          |
| 0         0           -(f+n)/(f-n)  -2fn/(f-n)|
| 0         0           -1          0          |
```

### Pimplパターン

`MaskGenerator` クラスはPimplイディオムを採用し、公開ヘッダにVulkan依存の型を含めない設計としている。

### Push Constants

MVP行列の渡し方はPush Constantsを使用。ディスクリプタセットの確保が不要で、オフスクリーンレンダリングに適した軽量な手法。

## コーディング規約

- Google C++ Style Guide に準拠
- Formatter: clang-format（`.clang-format` 参照）
- Linter: clang-tidy（`.clang-tidy` 参照）
- C++17

## Docker

```bash
# ビルド
docker compose build

# GPUなし環境（CPUフォールバック/Mesa llvmpipe）で実行する場合
docker compose run --rm maskgen /workspace/model/bun_zipper_res4.ply -o /workspace/output.png

# GPUを使用する場合（Intel iGPU等）
docker compose run --rm --device /dev/dri:/dev/dri maskgen \
  /workspace/model/bun_zipper_res4.ply -o /workspace/output.png
```

ホスト側でVulkanが利用可能か確認:

```bash
vulkaninfo --summary
```
