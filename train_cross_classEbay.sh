nohup python train_mx_ebay_margin.py --gpus=0 --batch-k=2 --batch-size=80 --epochs=55 --use_pretrained --use_viz --name=Ebay_Crossclass --data=EbayCrossClass >mytrainEbay_Crossclass.log 2>&1 &
