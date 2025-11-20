python main.py --num-samples 10000 --save-dir ./test4_10000/I-RAVEN

python visual.py --dataset ./test4_10000/I-RAVEN --num_vis 100 --save_dir ./visual/test4 \

# prb
now=$(date +"%Y-%m-%d-%H-%M-%S")
python -u /home/scxhc1/MNR_IJCAI25/cot_main.py --dataset-name I-RAVEN --dataset-dir /home/scxhc1/i-raven-cot/test4_10000 --gpu 0,1 --fp16 \
            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
            -a cot_prb_mask34 --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
            --classifier-hidreduce 4 --ckpt ckpts/cot_prb_ir_adam_${now}\
            --workers 2 --in-channels 1\
            2>&1 | tee log/cot/${now}_cotv3_prb_mask34.txt &
now=$(date +"%Y-%m-%d-%H-%M-%S")
python -u /home/scxhc1/MNR_IJCAI25/cot_main.py --dataset-name I-RAVEN --dataset-dir /home/scxhc1/i-raven-cot/test4_10000 --gpu 0,1 --fp16 \
            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
            -a cot_prb --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
            --classifier-hidreduce 4 --ckpt ckpts/cot_prb_ir_adam_${now}\
            --workers 2 --in-channels 1\
            2>&1 | tee log/cot/${now}_cotv3_prb_nomask.txt &
now=$(date +"%Y-%m-%d-%H-%M-%S")
python -u /home/scxhc1/MNR_IJCAI25/cot_main.py --dataset-name I-RAVEN --dataset-dir /home/scxhc1/i-raven-cot/test4_10000 --gpu 0,1 --fp16 \
            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
            -a cot_prb_mask4 --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
            --classifier-hidreduce 4 --ckpt ckpts/cot_prb_ir_adam_${now}\
            --workers 2 --in-channels 1\
            2>&1 | tee log/cot/${now}_cotv3_prb_mask4.txt &
now=$(date +"%Y-%m-%d-%H-%M-%S")
python -u /home/scxhc1/MNR_IJCAI25/cot_main.py --dataset-name I-RAVEN --dataset-dir /home/scxhc1/i-raven-cot/test4_10000 --gpu 0,1 --fp16 \
            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
            -a cot_prb_mask234 --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
            --classifier-hidreduce 4 --ckpt ckpts/cot_prb_ir_adam_${now}\
            --workers 2 --in-channels 1\
            2>&1 | tee log/cot/${now}_cotv3_prb_mask234.txt