seed=1
arch=resnet18
rewind_epoch=3
dataset=TinyImagenet
python -u main_imp.py \
	--data ../data \
	--dataset $dataset \
	--arch $arch \
	--seed $seed \
	--prune_type rewind_lt \
	--rewind_epoch $rewind_epoch \
	--save_dir "LT_rewind"$rewind_epoch"_"$dataset"_"$arch"_seed"$seed"" \
	--rate 0.2 \
	--pruning_times 1 \
	--epochs 160 \
	--batch_size 128 \
	--imagenet_arch	
