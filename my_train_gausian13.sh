for i in 3
do
   python mytrain.py --modelname=unet_test --batch_size=20 --Kfold=4 --activation=NONE --epochs=501 --use_train --knum=$i --gpu=0 --BCE --deleteall 
done

