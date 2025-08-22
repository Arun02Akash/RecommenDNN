Neural Collaborative Filtering (NCF)
====================================

This repository provides Keras implementations of the models proposed in:
He Xiangnan et al. Neural Collaborative Filtering. WWW 2017.

Models included:
- GMF (Generalized Matrix Factorization)
- MLP (Multi-Layer Perceptron for recommendation)
- NeuMF (Neural Matrix Factorization, combining GMF and MLP)


Environment
-----------
- Keras version: 1.0.7
- Theano version: 0.8.0

Note: These codes were originally implemented with Theano as the backend.


Running the Models
------------------

Each script includes a parse_args() function for command-line arguments.

Example commands:

Run GMF:
python GMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --regs [0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1

Run MLP:
python MLP.py --dataset ml-1m --epochs 20 --batch_size 256 --layers [64,32,16,8] --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1

Run NeuMF (without pre-training):
python NeuMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1

Run NeuMF (with pre-training):
python NeuMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1 --mf_pretrain Pretrain/ml-1m_GMF_8_1501651698.h5 --mlp_pretrain Pretrain/ml-1m_MLP_[64,32,16,8]_1501652038.h5

Tip:
If you are using zsh and encounter an error like:
zsh: no matches found: [64,32,16,8]
Use single quotes for array parameters:
--layers '[64,32,16,8]'


Docker Quickstart
-----------------

1. Install Docker Engine:
- Ubuntu: https://docs.docker.com/engine/installation/linux/ubuntu/
- MacOS: https://docs.docker.com/docker-for-mac/install/
- Windows: https://docs.docker.com/docker-for-windows/install/

2. Build Docker Image:
docker build --no-cache=true -t ncf-keras-theano .

3. Run with Docker:

Run GMF:
docker run --volume=$(pwd):/home ncf-keras-theano python GMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --regs [0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1

Run MLP:
docker run --volume=$(pwd):/home ncf-keras-theano python MLP.py --dataset ml-1m --epochs 20 --batch_size 256 --layers [64,32,16,8] --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1

Run NeuMF (without pre-training):
docker run --volume=$(pwd):/home ncf-keras-theano python NeuMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1

Run NeuMF (with pre-training):
docker run --volume=$(pwd):/home ncf-keras-theano python NeuMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1 --mf_pretrain Pretrain/ml-1m_GMF_8_1501651698.h5 --mlp_pretrain Pretrain/ml-1m_MLP_[64,32,16,8]_1501652038.h5


Dataset
-------

Provided datasets:
- MovieLens 1M (ml-1m)
- Pinterest (pinterest-20)

Files:
- train.rating: Training instances (userID, itemID, rating, timestamp if available)
- test.rating: Testing instances (positive only, one per user)
- test.negative: For each test rating, contains 99 sampled negative items in the format:
  (userID,itemID) \t negItem1 \t negItem2 ...


Notes on NeuMF Tuning
---------------------
- For small embedding dimensions: training NeuMF without pre-training can outperform GMF and MLP.
- For larger embedding dimensions: pre-training GMF and MLP before running NeuMF can yield better performance.
- Regularization tuning may be necessary.


Last Update
-----------
December 23, 2018
