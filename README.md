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


- python main.py --lidar 0 --metric q_ngr --model_type fc --model_name big --model_path pgd/models/ --mnist_res_path res/q_ngr/ 
- python main.py --lidar 0 --metric q_inv --model_type fc --model_name big --model_path pgd/models/ --mnist_res_path res/q_inv/ 
- python main.py --lidar 0 --metric q_exp --model_type fc --model_name big --model_path pgd/models/ --mnist_res_path res/q_exp/ 

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

- python main.py --image 1 --cifar small --lidar 0 --metric q_ngr --model_type cnn --model_name adv --model_path CNN/models/ --mnist_res_path res/q_ngr/ --dataset cifar

- python main.py --image 1 --cifar big --lidar 0 --metric q_inv --model_type cnn --model_name adv --model_path CNN/models/ --mnist_res_path res/q_inv/ --dataset cifar

- python main.py --image 1 --cifar small --lidar 0 --metric q_ngr --model_type cnn --model_name adv --model_path CNN/models/ --mnist_res_path res/q_ngr/ --dataset cifar

- python main.py --image 1 --cifar small --lidar 0 --metric q_inv --model_type cnn --model_name adv --model_path CNN/models/ --mnist_res_path res/q_inv/ --dataset cifar


### Slope and Fraction Calculation

- python test.py --lidar 0 --metric q_exp --model_type fc --model_name big --model_path pgd/models/ --mnist_data_path res/q_exp/ --mnist_res_path statistics/q_exp/ 

- python test.py --lidar 0 --metric q_ngr --model_type fc --model_name ori --model_path pgd/models/ --mnist_data_path res/q_ngr/ --mnist_res_path statistics/q_ngr/

- python test.py --lidar 0 --metric q_inv --model_type fc --model_name ori --model_path pgd/models/ --mnist_data_path res/q_inv/ --mnist_res_path statistics/q_inv/

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
- python test.py --lidar 0 --metric q_exp --model_type fc_linear --model_name ori --model_path pgd/models/ --mnist_data_path res/q_exp/ --mnist_res_path statistics/q_exp/



## LiDAR

- python main.py --mnist 0 --metric q_inv --lidar_res_path res/lidar/

- python test.py --mnist 0 --metric q_inv --lidar_data_path res/lidar/ --lidar_res_path statistics/lidar/