# Graph_Curvature_NN

## Neural Data Graph Weight Definitions

- q_NGR

- q_INV

- q_EXP

## MNIST

### FC

- python main.py --lidar 0 --metric q_ngr --model_type fc --model_name adv --model_path pgd/models/ --mnist_res_path res/q_ngr/
- python main.py --lidar 0 --metric q_inv --model_type fc --model_name adv --model_path pgd/models/ --mnist_res_path res/q_inv/
- python main.py --lidar 0 --metric q_exp --model_type fc --model_name adv --model_path pgd/models/ --mnist_res_path res/q_exp/


- python main.py --lidar 0 --metric q_ngr --model_type fc --model_name big_adv --model_path pgd/models/ --mnist_res_path res/q_ngr/ 
- python main.py --lidar 0 --metric q_inv --model_type fc --model_name big_ori --model_path pgd/models/ --mnist_res_path res/q_inv/ 
- python main.py --lidar 0 --metric q_exp --model_type fc --model_name big_adv --model_path pgd/models/ --mnist_res_path res/q_exp/ 

### FC Linear

- python main.py --lidar 0 --metric q_ngr --model_type fc_linear --model_name ori --model_path pgd/models/ --mnist_res_path res/q_ngr/
- python main.py --lidar 0 --metric q_inv --model_type fc_linear --model_name ori --model_path pgd/models/ --mnist_res_path res/q_inv/
- python main.py --lidar 0 --metric q_exp --model_type fc_linear --model_name ori --model_path pgd/models/ --mnist_res_path res/q_exp/


### CNN

- python main.py --lidar 0 --metric q_ngr --model_type cnn --model_name ori --model_path CNN/models/ --mnist_res_path res/q_ngr/
- python main.py --lidar 0 --metric q_inv --model_type cnn --model_name ori --model_path CNN/models/ --mnist_res_path res/q_inv/
- python main.py --lidar 0 --metric q_exp --model_type cnn --model_name ori --model_path CNN/models/ --mnist_res_path res/q_exp/


## CIFAR10

- python main.py --image 1 --cifar small --lidar 0 --metric q_inv --model_type cnn --model_name ori --model_path CNN/models/ --mnist_res_path res/q_inv/ --dataset cifar

- python main.py --image 1 --cifar small --lidar 0 --metric q_ngr --model_type cnn --model_name ori --model_path CNN/models/ --mnist_res_path res/q_ngr/ --dataset cifar

- python main.py --image 1 --cifar big --lidar 0 --metric q_inv --model_type cnn --model_name adv --model_path CNN/models/ --mnist_res_path res/q_inv/ --dataset cifar

- python main.py --image 1 --cifar small --lidar 0 --metric q_ngr --model_type cnn --model_name adv --model_path CNN/models/ --mnist_res_path res/q_ngr/ --dataset cifar

- python main.py --image 1 --cifar small --lidar 0 --metric q_inv --model_type cnn --model_name adv --model_path CNN/models/ --mnist_res_path res/q_inv/ --dataset cifar


### Slope and Fraction Calculation

- python test.py --lidar 0 --metric q_ngr --model_type fc --model_name big_adv --model_path pgd/models/ --mnist_data_path res/q_ngr/ --mnist_res_path statistics/q_ngr/ 

- python test.py --lidar 0 --metric q_inv --model_type fc --model_name big_adv --model_path pgd/models/ --mnist_data_path res/q_inv/ --mnist_res_path statistics/q_inv/ 

- python test.py --lidar 0 --metric q_ngr --model_type fc --model_name adv --model_path pgd/models/ --mnist_data_path res/q_ngr/ --mnist_res_path statistics/q_ngr/

- python test.py --lidar 0 --metric q_inv --model_type fc --model_name adv --model_path pgd/models/ --mnist_data_path res/q_inv/ --mnist_res_path statistics/q_inv/

- python test.py --lidar 0 --metric q_exp --model_type fc --model_name ori --model_path pgd/models/ --mnist_data_path res/q_exp/ --mnist_res_path statistics/q_exp/

- python test.py --lidar 0 --metric q_ngr --model_type cnn --model_name ori --model_path CNN/models/ --mnist_data_path res/q_ngr/ --mnist_res_path statistics/q_ngr/
- python test.py --lidar 0 --metric q_inv --model_type cnn --model_name ori --model_path CNN/models/ --mnist_data_path res/q_inv/ --mnist_res_path statistics/q_inv/
- python test.py --lidar 0 --metric q_exp --model_type cnn --model_name ori --model_path CNN/models/ --mnist_data_path res/q_exp/ --mnist_res_path statistics/q_exp/

- python test.py --image 1 --cifar small --lidar 0 --metric q_inv --model_type cnn --model_name adv --model_path CNN/models/ --mnist_data_path res/q_inv/ --mnist_res_path statistics/q_inv/ --dataset cifar

- python test.py --image 1 --cifar small --lidar 0 --metric q_inv --model_type cnn --model_name adv --model_path CNN/models/ --mnist_data_path res/q_inv/ --mnist_res_path statistics/q_inv/ --dataset cifar


### CIFAR test

- python test.py --image 1 --cifar small --lidar 0 --metric q_inv --model_type cnn --model_name ori --model_path CNN/models/ --mnist_data_path res/q_inv/ --mnist_res_path statistics/q_inv/ --dataset cifar

- python test.py --image 1 --cifar small --lidar 0 --metric q_inv --model_type cnn --model_name adv --model_path CNN/models/ --mnist_data_path res/q_inv/ --mnist_res_path statistics/q_inv/ --dataset cifar

- python test.py --image 1 --cifar big --lidar 0 --metric q_inv --model_type cnn --model_name adv --model_path CNN/models/ --mnist_data_path res/q_inv/ --mnist_res_path statistics/q_inv/ --dataset cifar


### LiDAR test

- python test.py --lidar 0 --metric q_ngr --model_type fc_linear --model_name ori --model_path pgd/models/ --mnist_data_path res/q_ngr/ --mnist_res_path statistics/q_ngr/
- python test.py --lidar 0 --metric q_inv --model_type fc_linear --model_name ori --model_path pgd/models/ --mnist_data_path res/q_inv/ --mnist_res_path statistics/q_inv/
- python test.py --lidar 0 --metric w3 --model_type fc --model_name ori --model_path pgd/models/ --mnist_data_path res/q_exp/ --mnist_res_path statistics/q_exp/



## LiDAR

- python main.py --image 0 --lidar 1 --metric w3 --lidar_res_path res/lidar/w3/ --model_path Lidar/models/ --sample_num 50

- python test.py --image 0 --lidar 1 --metric q_inv --lidar_data_path res/lidar/ --lidar_res_path statistics/lidar/ 



## Removal

- python removal.py --lidar 0 --metric q_inv --model_type fc --model_name big_ori --model_path pgd/models/ --mnist_res_path res/q_inv/ --edge 1 --node 0

- python removal.py --lidar 0 --metric q_inv --model_type fc --model_name adv --model_path pgd/models/ --mnist_res_path res/q_inv/ --edge 0 --node 1

- python removal.py --lidar 0 --metric q_inv --model_type cnn --model_name ori --model_path CNN/models/ --mnist_res_path res/cnn_ori_edges/ --edge 1 --node 0

- python removal.py --lidar 0 --image 1 --cifar small --metric q_inv --model_type cnn --model_name ori --model_path CNN/models/ --mnist_res_path res/edge_cifarori/ --dataset cifar --edge 1 --node 0

- python removal.py --lidar 0 --metric w1 --model_type fc --model_name ori --model_path pgd/models/ --mnist_res_path res/community/ --community 1

- python removal.py --lidar 0 --metric w1 --model_type cnn --model_name ori --model_path CNN/models/ --mnist_res_path res/community/ --community 1

- python removal.py --lidar 0 --image 1 --cifar small --metric w3 --model_type cnn --model_name ori --model_path CNN/models/ --mnist_res_path res/community/cifar_ori/ --dataset cifar --community 1

- python test.py --lidar 0 --image 1 --metric w3 --model_type fc --model_name ori --community 1 --mnist_data_path res/community0/ --mnist_res_path statistics/q_inv/

- python removal.py --lidar 0 --metric w3 --model_type fc --model_name big_ori --model_path pgd/models/new/ --mnist_data_path res/w3_with0/tanh/bigori/ --mnist_res_path statistics/remove_fre/tanh/bigori/ --edge 1 --node 0 --activation tanh

- python test.py --lidar 0 --image 1 --metric w3 --model_type fc --model_name big_ori --community 0 --mnist_data_path res/w3_with0/relu/bigori/ --mnist_res_path statistics/w3_with0_community/relu/bigori/

- python test.py --lidar 0 --image 0 --metric w3 --model_type cnn --model_name adv --community 1 --mnist_data_path res/w3_with0/relu/cnnadv/ --mnist_res_path statistics/indegree/relu/cnnadv/

- python removal.py --lidar 0 --metric w3 --model_type fc --model_name big_ori --model_path pgd/models/new/ --mnist_res_path statistics/w3/tanh/bigori/ --edge 0 --node 0 --activation tanh --community 1 --sample_num 100

- python removal.py --lidar 0 --metric w3 --model_type fc --model_name big_ori --model_path pgd/models/new/ --mnist_data_path res/w3/relu/ori1/ --mnist_res_path statistics/remove_new/relu/bigori/ --edge 1 --node 0 --activation relu --sample_num 10

- python removal.py --lidar 0 --metric w3 --model_type cnn --model_name ori --model_path CNN/models/new/ --mnist_data_path res/w3/relu/cifarori/ --mnist_res_path statistics/remove/relu/cifarori/ --edge 1 --node 0 --activation relu --sample_num 10 --dataset cifar

- python removal.py --lidar 1 --image 0 --metric w4 --model_type DDPG --model_name 64 --model_path Lidar/models/ --mnist_data_path res/lidar/w4/ --mnist_res_path statistics/remove/lidar/ddpg64/ --edge 1 --sample_num 10

- python removal.py --lidar 0 --metric w4 --model_type fc --model_name big_wd --model_path pgd/models/new/ --mnist_res_path res/w4/relu/bigwd/ --edge 0 --node 0 --activation relu --community 1 --sample_num 100

- python removal.py --lidar 0 --metric w4 --model_type cnn --model_name ori --model_path CNN/models/new/ --mnist_res_path res/w4/tanh/cnnori/ --edge 0 --node 0 --activation tanh --community 1 --sample_num 100

- python removal.py --lidar 0 --image 1 --cifar small --metric w4 --model_type cnn --model_name ori --model_path CNN/models/new/ --mnist_res_path res/w4/tanh/cifarori/ --dataset cifar --community 1 --activation tanh --sample_num 100