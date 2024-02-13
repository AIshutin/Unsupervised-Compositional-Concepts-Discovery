model_path="stabilityai/stable-diffusion-2-1-base"
train_data_dir="../subject-driven-gen-eval/datasets/dreambooth"
placeholder_tokens="<t1>,<t2>"
class_folder_names="can" 
learnable_property="object"
output_dir=out-dir3

DEVICE=$CUDA_VISIBLE_DEVICES
python create_accelerate_config.py --gpu_id "${DEVICE}"

accelerate env

accelerate launch --config_file accelerate_config.yaml main.py \
--pretrained_model_name_or_path "${model_path}" \
--train_data_dir ${train_data_dir}  \
--placeholder_tokens ${placeholder_tokens} \
--resolution=512  --class_folder_names ${class_folder_names} \
--train_batch_size=2 --gradient_accumulation_steps=8 --repeats 1 \
--learning_rate=5.0e-03 --scale_lr --lr_scheduler="constant" --max_train_steps 3000 \
--lr_warmup_steps=0   --output_dir ${output_dir} \
--learnable_property "${learnable_property}"  \
--checkpointing_steps 360 --mse_coeff 1 --seed 0 \
--add_weight_per_score \
--use_conj_score --init_weight 5 \
--validation_step 360 \
--report_to wandb \
--num_iters_per_image 120 --num_images_per_class 5

python inference.py --model_path ${output_dir} --prompts "a photo of <t1>" "a photo of <t2>" --num_images 64 --bsz 8