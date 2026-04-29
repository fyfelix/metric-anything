# HAMMER 评估适配说明

本目录将导出的 HAMMER 评估 pipeline 适配到当前 MetricAnything 项目。
当前适配是固定目标，不是通用模型注册框架：

- 模型：`models/student_depthmap/depth_model.py::MetricAnythingDepthMap`
- 默认 checkpoint / repo id：`yjh001/metricanything_student_depthmap`
- 数据集：HAMMER JSONL
- 指标：沿用 `utils/metric.py`，未修改指标定义

当前模型是 RGB-only，并使用焦距感知的 metric depth 推理。为了兼容原
`HAMMERDataset`，脚本仍会读取 HAMMER raw depth 路径字段，但 `infer.py`
不会把 raw depth 送入模型。

## 文件说明

- `run_eval.sh`：主入口，先推理保存预测，再运行指标评估。
- `infer.py`：加载 MetricAnything Student DepthMap，并为每个样本写出
  `*.npy` 预测。
- `eval.py`：从原 pipeline 复制；读取 `*.npy`、GT depth 和 valid mask，
  写出 CSV/JSON 指标。
- `dataset.py`：从原 pipeline 复制的 HAMMER JSONL loader。
- `utils/metric.py`：从原 pipeline 复制的固定指标实现。

## 数据集路径

默认数据集路径为：

```bash
data/HAMMER/test.jsonl
```

如需覆盖：

```bash
DATASET_PATH=/path/to/HAMMER/test.jsonl ./evaluation/run_eval.sh
```

JSONL 样本需保留原 HAMMER 字段：`rgb`、`d435_depth`、`l515_depth`、
`tof_depth`、`depth` 和 `depth-range`。

## 运行方式

使用默认 Hugging Face 模型：

```bash
./evaluation/run_eval.sh
```

使用本地 checkpoint 或其他兼容 repo id：

```bash
./evaluation/run_eval.sh /path/to/student_depthmap.pt d435 false
```

参数格式：

```text
./evaluation/run_eval.sh [model_ref] [raw_type=d435] [cleanup_npy=false]
```

常用环境变量：

```text
DATASET_PATH          默认：data/HAMMER/test.jsonl
OUTPUT_DIR            默认：evaluation_outputs/hammer_<model>_data_<raw_type>
BATCH_SIZE            默认：1；建议当前官方推理路径使用 1
NUM_WORKERS           默认：0
DEVICE                例如 cuda、cuda:0、cpu 或 mps
F_PX                  可选，所有图像统一使用的焦距像素值
INTRINSICS_PATH       相机内参 txt；默认：<DATASET_PATH 所在目录>/intrinsics.txt
REQUIRE_FOCAL         true 表示缺少显式 focal 时直接报错
LOCAL_FILES_ONLY      true 表示禁止 Hugging Face 下载，只使用本地缓存
SAVE_VIS              true 表示额外保存 depth 预览 PNG
PYTHON_BIN            默认：python3
```

## 预测输出格式

`infer.py` 会在 `OUTPUT_DIR` 中为每个 HAMMER 样本保存一个
`<scene>#<stem>.npy`。每个文件都是 `HxW float32` metric depth，单位为
meter，并由 `MetricAnythingDepthMap.infer()` resize 回原始 RGB 分辨率。

模型 API 已直接返回 metric depth，因此不需要在本适配中执行 disparity 或
inverse depth 到 depth 的转换。

焦距解析顺序：

1. `F_PX` / `--f-px` 显式覆盖。
2. `INTRINSICS_PATH` / `--intrinsics-path` 指向的 3x3 内参 txt，默认是
   `data/HAMMER/intrinsics.txt`；使用第一行第一列 `fx = K[0,0]` 作为 `f_px`。
3. HAMMER JSONL 样本中的 focal 字段，例如 `f_px`、`fx`、`cam_in`、
   `intrinsics`、`camera_intrinsics` 或 `K`。
4. RGB 图像旁边的同名 JSON，格式为 `{"cam_in": [fx, fy, cx, cy]}`。
5. 回退使用 RGB 图像宽度。

推理结束后，`infer.py` 会在输出目录写出 `focal_sources.json`，记录本次评估
中 focal 来源的样本数量。若希望缺少真实 focal 时直接报错，可以直接运行
`evaluation/infer.py` 并加上 `--require-focal`，或设置 `REQUIRE_FOCAL=true`。

默认不启用、也不需要 alignment。若后续评估 relative-depth checkpoint，
应增加显式 alignment 选项，并在保存预测和计算指标前说明预测已对齐。

## 已知限制

- 本适配只面向 `MetricAnythingDepthMap`，不是通用模型 registry。
- `raw_type` 只决定 HAMMER raw depth 字段路径，不影响 RGB-only 模型推理。
- 如果默认 Hugging Face checkpoint 未在本地缓存，首次加载需要网络访问。
