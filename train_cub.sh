nohup python train_mx_ebay_margin.py --gpus=0 --batch-k=5 --batch-size=70 --use_pretrained --use_viz --epochs=30 --name=CUB_200_2011 --data=CUB_200_2011 >mytraincub200.log 2>&1 &
