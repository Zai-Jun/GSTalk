dataset=$1
workspace=$2
gpu_id=$3
#audio_extractor='ave' # deepspeech, esperanto, hubert
audio_extractor='deepspeech' # deepspeech, esperanto, hubert


pretrain_project_path="out/eyetri"

pretrain_face_path=${pretrain_project_path}/chkpnt_ema_face_latest.pth
pretrain_mouth_path=${pretrain_project_path}/chkpnt_ema_mouth_latest.pth

# n_views=500 # 20s
# n_views=250 # 10s
n_views=125 # 5s


export CUDA_VISIBLE_DEVICES=$gpu_id

python train.py --type face -s $dataset -m $workspace --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor $audio_extractor --pretrain_path $pretrain_face_path --iterations 14000 --sh_degree 1 --N_views $n_views

python synthesize_fuse.py -s $dataset -m $workspace --eval --audio_extractor $audio_extractor --dilate

python metrics.py $workspace/test/ours_None/renders/out.mp4 $workspace/test/ours_None/gt/out.mp4
