# Nomenclature
There are some naming differences which are an artifact of our iterative development. We will first outline how the names in the codebase translate to the names in the paper:
1. *FedAvgBatch = FedAvg*: the hyperparameter beta corresponds to eta
2. *Naive = Local Training*: the hyperparameter beta corresponds to alpha
3. *FTFA = PerAvgEpochBatch*: the hyperparameter beta, learning_rate corresponds to eta, alpha respectively
4. *RTFA = PerRidgeEpochBatch*: the hyperparameter beta, learning_rate, user_ridge_penalty corresponds to eta, alpha, lambda respectively
5. *MAML-FL-HF = PerAvgHF*: the hyperparameter beta, learning_rate, delta_HF corresponds to eta, alpha, delta respectively
6. *MAML-FL-FO = PerAvg*: the hyperparameter beta, learning_rate corresponds to eta, alpha respectively

The *PerAvgHFOneStep*, *PerAvgOneStep*, and *PerAvgEpochBatchOneStep* algorithms correspond to the *PerAvgHF*, *PerAvg*, and *PerAvgEpochBatch* algorithms with one batch of personalization.


# Example workflow for {Dataset Name}:
0. Have conda installed; run the following command to install the packages necessary for running the code ```conda env create -f environment.yaml``` inside the fed-learn-cuda11 folder. run ```conda activate fed-learn-cuda11``` to activate the environment.
1. Download data from by navigating to ./preprocess_and_pretrain/FedML/data/{Dataset Name}/ Then run ```sh download_{Dataset Name}.sh```
2. Create dataset for preprocessing by running ``` python {Dataset Name}_make_preprocessing_data.py``` in the ./preprocess_and_pretrain/ folder
3. Run preprocessing by running ``` python {Dataset Name}_pretrain_main.py``` in the ./preprocess_and_pretrain/ folder
4. Create preprocessed datasets to be fed into last layer net federated learning algorithms by running ``` python {Dataset Name}_make_preprocessed_datasets.py``` in the ./preprocess_and_pretrain/ folder
5. In the fed-learn-code folder, run the following with [...] replaced with the desired hyperparameter

```
python main.py --algorithm=[...] --dataset=[...] --model=[...] --beta=[...] --learning_rate=[...] --user_ridge_penalty=[...] --delta_HF=[...] --datasetnumber=[...] --seed=[...] ---no-cuda=[...] --decimate=50 --batch_size=32 --local_epochs=20  --num_global_iters=401 --numusers=20 --optimizer=SGD --rank=0 --times=1 --personal_epochs=10
```

The algorithm options are ["pFedMe", "PerAvg", "FedAvgBatch", "PerAvgEpochBatch", "PerAvgHF", "Naive", "PerRidgeEpochBatch", "PerAvgOneStep", "PerAvgEpochBatchOneStep", "PerAvgHFOneStep"]

The dataset options are ["prep_FederatedEMNIST", "prep_Femnist", "prep_Shakespeare", "prep_CIFAR100", "prep_Stackoverflownwp"]

The model options are ["cnn", "rnn", "resnet", "rnnstackoverflow"].

no-cuda specifies whether to use gpu for training. All of our testing was on gpu.

See fed-learn-code/main.py for details on each option. Checkout the P2-sweeps folder to see what choices of hyperparameters we searched over. Checkout P3-seed-sweeps and P4-dataset-sweeps to find the best parameters. 
