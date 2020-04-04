# Setup

```bash
sh fetch_data_and_setup.sh
conda env create -f environment.yml
conda activate clvision-challenge
sh create_submission.sh
```

# Run

```bash
python main.py     --scenario="multi-task-nc" --epochs="5" --sub_dir="baseline"
python main_ewc.py --scenario="multi-task-nc" --epochs="5" --sub_dir="ewc"
```
