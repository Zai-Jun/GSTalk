dataset=$1
workspace=$2
gpu_id=0
audio_extractor='deepspeech' # deepspeech, esperanto, hubert

export CUDA_VISIBLE_DEVICES=$gpu_id

python pretrain.py -s $dataset -m $workspace --type face --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor $audio_extractor --iterations 50000
