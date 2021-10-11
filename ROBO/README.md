# ROBO
Save data to `../input` and then run the command below to generate model weights:

```
python train_mlp_controller.py bumblebee
python train_mlp_controller.py beetle
python train_mlp_controller.py butterfly
```

The scripts were executed with torch 1.9.0 and sklearn 0.24.2 but older version may also work.

once the model weights are generated (we have also provided the weights in the folder), you can zip `model` and submit it.