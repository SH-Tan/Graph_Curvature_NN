# Graph_Curvature_NN
## Parameters ((specific details can be found in removal.py file))

- community: 1 means for neural curvature calculation

- edge: 1 means for edge removal experiments 

- metric: w4 used for this paper

- model type: cnn used for this paper

- model_name: ori (CE), wd (WD), adv (AT) (CIFAR-100 use ori)

- model_path: path to trained models

- mnist_res_path: results saved path

- mnist_data_path: save results used for removal experiments

- activation: relu or tanh

- sample_num: number of examples used for neural graph calculation

## Neural Data Graph Weight Definitions

- w4

## MNIST

- python removal.py --lidar 0 --metric w4 --model_type cnn --model_name ori --model_path CNN/models/new/ --mnist_res_path res/w4/tanh/cnnori/ --edge 0 --node 0 --activation tanh --community 1 --sample_num 100

- python removal.py --lidar 0 --metric w4 --model_type cnn --model_name ori --model_path CNN/models/new/ --mnist_data_path res/w4/relu/cifarori/ --mnist_res_path statistics/relu/cnnori/ --edge 1 --node 0 --activation relu --sample_num 100


## CIFAR10

- python removal.py --lidar 0 --metric w4 --model_type cnn --model_name ori --model_path CNN/models/new/ --mnist_res_path res/w4/tanh/cifarori/ --edge 0 --node 0 --activation tanh --community 1 --sample_num 100 --dataset cifar10

- python removal.py --lidar 0 --metric w4 --model_type cnn --model_name ori --model_path CNN/models/new/ --mnist_data_path res/w4/relu/cifarori/ --mnist_res_path statistics/aaai/relu/cifarori/ --edge 1 --node 0 --activation relu --sample_num 100 --dataset cifar10

## CIFAR100

- python removal.py --lidar 0 --metric w4 --model_type cnn --model_name ori --model_path CNN/models/new/ --mnist_res_path res/w4/tanh/cifarori/ --edge 0 --node 0 --activation tanh --community 1 --sample_num 100 --dataset cifar100

- python removal.py --lidar 0 --metric w4 --model_type cnn --model_name ori --model_path CNN/models/new/ --mnist_data_path res/w4/relu/cifarori/ --mnist_res_path statistics/aaai/relu/cifarori/ --edge 1 --node 0 --activation relu --sample_num 100 --dataset cifar100
