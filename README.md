# Graph_Curvature_NN

## Neural Data Graph Weight Definitions

- q_NGR

- q_INV

- q_EXP

## MNIST

### FC

- python main.py --metric q_ngr --model_type fc --model_name ori --model_path pgd/models/ --res_path res/q_ngr/
- python main.py --metric q_inv --model_type fc --model_name ori --model_path pgd/models/ --res_path res/q_inv/
- python main.py --metric q_exp --model_type fc --model_name ori --model_path pgd/models/ --res_path res/q_exp/

### FC Linear

- python main.py --metric q_ngr --model_type fc_linear --model_name ori --model_path pgd/models/ --res_path res/q_ngr/
- python main.py --metric q_inv --model_type fc_linear --model_name ori --model_path pgd/models/ --res_path res/q_inv/
- python main.py --metric q_exp --model_type fc_linear --model_name ori --model_path pgd/models/ --res_path res/q_exp/


### CNN

- python main.py --metric q_ngr --model_type cnn --model_name ori --model_path CNN/models/ --res_path res/q_ngr/
- python main.py --metric q_inv --model_type cnn --model_name ori --model_path CNN/models/ --res_path res/q_inv/
- python main.py --metric q_exp --model_type cnn --model_name ori --model_path CNN/models/ --res_path res/q_exp/

### Slope and Fraction Calculation

- python test.py --metric q_ngr --model_type fc --model_name ori --model_path pgd/models/ --data_path res/q_ngr/ --res_path statistics/q_ngr/

- python test.py --metric q_inv --model_type fc --model_name ori --model_path pgd/models/ --data_path res/q_inv/ --res_path statistics/q_inv/

- python test.py --metric q_exp --model_type fc --model_name ori --model_path pgd/models/ --data_path res/q_exp/ --res_path statistics/q_exp/

- python test.py --metric q_ngr --model_type cnn --model_name ori --model_path CNN/models/ --data_path res/q_ngr/ --res_path statistics/q_ngr/
- python test.py --metric q_inv --model_type cnn --model_name ori --model_path CNN/models/ --data_path res/q_inv/ --res_path statistics/q_inv/
- python test.py --metric q_exp --model_type cnn --model_name ori --model_path CNN/models/ --data_path res/q_exp/ --res_path statistics/q_exp/


- python test.py --metric q_ngr --model_type fc_linear --model_name ori --model_path pgd/models/ --data_path res/q_ngr/ --res_path statistics/q_ngr/
- python test.py --metric q_inv --model_type fc_linear --model_name ori --model_path pgd/models/ --data_path res/q_inv/ --res_path statistics/q_inv/
- python test.py --metric q_exp --model_type fc_linear --model_name ori --model_path pgd/models/ --data_path res/q_exp/ --res_path statistics/q_exp/



## LiDAR