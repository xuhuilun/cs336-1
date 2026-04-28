

---

# 🔥 一行命令版消融实验（直接复制）
## 1. 默认配置（RMS ON + RoPE ON + PreNorm + SwiGLU）✅️
```bash
PYTHONPATH=. uv run python cs336_basics/main_train.py \
  --train_data_path data/TinyStoriesV2-GPT4-train.bin \
  --valid_data_path data/TinyStoriesV2-GPT4-valid.bin \
  --device cuda \
  --out_dir model_result/TinyStories_baseline \
  --run_name "baseline"
```

## 2. 关闭 RMSNorm
```bash
PYTHONPATH=. uv run python cs336_basics/main_train.py \
  --train_data_path data/TinyStoriesV2-GPT4-train.bin \
  --valid_data_path data/TinyStoriesV2-GPT4-valid.bin \
  --device cuda \
  --no_rms_norm \
  --out_dir model_result/TinyStories_no_rms \
  --run_name "no_rms_norm"
```

## 3. 使用 PostNorm✅️
```bash
PYTHONPATH=. uv run python cs336_basics/main_train.py \
  --train_data_path data/TinyStoriesV2-GPT4-train.bin \
  --valid_data_path data/TinyStoriesV2-GPT4-valid.bin \
  --device cuda \
  --norm_mode post \
  --out_dir model_result/TinyStories_post_norm \
  --run_name "post_norm"
```

## 4. 关闭 RoPE
```bash
PYTHONPATH=. uv run python cs336_basics/main_train.py \
  --train_data_path data/TinyStoriesV2-GPT4-train.bin \
  --valid_data_path data/TinyStoriesV2-GPT4-valid.bin \
  --device cuda \
  --no_rope \
  --out_dir model_result/TinyStories_no_rope \
  --run_name "no_rope"
```

## 5. 使用 SiLU（不使用 SwiGLU）
```bash
PYTHONPATH=. uv run python cs336_basics/main_train.py \
  --train_data_path data/TinyStoriesV2-GPT4-train.bin \
  --valid_data_path data/TinyStoriesV2-GPT4-valid.bin \
  --device cuda \
  --ffn_type silu \
  --out_dir model_result/TinyStories_silu \
  --run_name "silu"
```

---

# 📌 超短总结（记这个就够）
- **默认**：全部开启（最强）
- `--no_rms_norm`：关归一化
- `--norm_mode post`：后归一化
- `--no_rope`：关位置编码
- `--ffn_type silu`：普通前馈层



