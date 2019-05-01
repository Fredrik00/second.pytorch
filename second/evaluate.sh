python ./pytorch/train.py evaluate --config_path=./configs/$1.config --model_dir=/notebooks/second_models/$2 --measure_time=True --batch_size=1 --eval_all=True 2>&1 | tee $2_results.txt
