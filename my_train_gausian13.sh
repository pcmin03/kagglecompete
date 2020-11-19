for i in 7
do
   python mytrain.py --modelname=unet_test --batch_size=20 --activation=NONE --epochs=501 --use_train --knum=$i --gpu=2 --BCE --deleteall 
done

