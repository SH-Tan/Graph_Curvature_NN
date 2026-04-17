# Code for Analyzing Neural Network using Graph Curvature

## Updated: Llama3 8b

## Image classification: Neural Graph Definition (Default: `w4`)

- `--metric w4`

---

## Example Command

(See `run_vgg.sh` for a full example)

```bash
python removal.py --image 1 --metric w4 --model_type cnn --model_name ori \
--model_path CNN/models/cnn/ \
--img_res_path res/w4/relu/cifarori_vgg9_new_curv0/ \
--dataset cifar10 \
--community 1 \
--edge 0 \
--activation relu \
--sample_num 1 \
--img_data_path res/w4/relu/cifarori_vgg9/ \
--alpha 0.9
```

---

## Main Script

- `removal.py` – main pruning script

---

## Arguments

| Argument | Description |
|--------|-------------|
| `--image` | Image classifier mode |
| `--metric` | Neural graph metric (default: `w4`) |
| `--model_name` | Training method: `ori` (CE), `wd` (Weight Decay), `adv` (Adversarial Training) |
| `--model_path` | Path to pretrained model |
| `--img_res_path` | Path to save pruning results |
| `--dataset` | Dataset name (e.g., `cifar10`) |
| `--community` | `1`: calculate graph curvature |
| `--edge` | `1`: count results and start pruning |
| `--activation` | Activation function (`relu` or `tanh`) |
| `--sample_num` | Number of calibration samples per class used for curvature calculation |
| `--img_data_path` | Path to curvature results used for pruning |
| `--alpha` | Parameter for curvature calculation (Definition 5 in the paper) |

