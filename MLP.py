import argparse
import numpy as np
from time import time
import multiprocessing as mp

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop

from Dataset import Dataset
from evaluate import evaluate_model


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP for recommendation.")
    parser.add_argument('--path', type=str, default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', type=str, default='ml-1m',
                        help='Dataset name.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', type=str, default='[64,32,16,8]',
                        help='List of layer sizes, e.g. [64,32,16,8]. '
                             'First layer size/2 = embedding dimension.')
    parser.add_argument('--reg_layers', type=str, default='[0,0,0,0]',
                        help='List of regularization values per layer.')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative samples per positive.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', type=str, default='adam',
                        help='Optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X epochs.')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()


#################### Model ####################
def get_model(num_users: int, num_items: int, layers=None, reg_layers=None) -> Model:
    """
    Build the MLP model for recommendation.
    """
    if layers is None:
        layers = [64, 32, 16, 8]
    if reg_layers is None:
        reg_layers = [0] * len(layers)
    assert len(layers) == len(reg_layers)

    # Inputs
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # Embedding layers
    embedding_dim = layers[0] // 2
    user_embedding = Embedding(
        input_dim=num_users,
        output_dim=embedding_dim,
        name='user_embedding',
        embeddings_initializer="random_normal",
        embeddings_regularizer=l2(reg_layers[0])
    )(user_input)

    item_embedding = Embedding(
        input_dim=num_items,
        output_dim=embedding_dim,
        name='item_embedding',
        embeddings_initializer="random_normal",
        embeddings_regularizer=l2(reg_layers[0])
    )(item_input)

    # Flatten embeddings
    user_latent = Flatten()(user_embedding)
    item_latent = Flatten()(item_embedding)

    # Concatenate user and item embeddings
    vector = Concatenate()([user_latent, item_latent])

    # Hidden layers
    for idx in range(1, len(layers)):
        vector = Dense(
            layers[idx],
            activation='relu',
            kernel_regularizer=l2(reg_layers[idx]),
            name=f'layer{idx}'
        )(vector)

    # Prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer="lecun_uniform", name='prediction')(vector)

    return Model(inputs=[user_input, item_input], outputs=prediction)


#################### Training Instances ####################
def get_train_instances(train, num_negatives: int, num_items: int):
    """
    Generate positive and negative training instances.
    """
    user_input, item_input, labels = [], [], []
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for _ in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return np.array(user_input), np.array(item_input), np.array(labels)


#################### Main ####################
if __name__ == '__main__':
    args = parse_args()
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)

    topK = 10
    evaluation_threads = mp.cpu_count()

    print("MLP arguments:", args)
    model_out_file = f'Pretrain/{args.dataset}_MLP_{args.layers}_{int(time())}.h5'

    # Load data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.train_matrix, dataset.test_ratings, dataset.test_negatives
    num_users, num_items = train.shape
    print(f"Load data done [{time()-t1:.1f} s]. "
          f"#user={num_users}, #item={num_items}, #train={train.nnz}, #test={len(testRatings)}")

    # Build model
    model = get_model(num_users, num_items, layers, reg_layers)

    # Compile
    optimizer_dict = {
        "adagrad": Adagrad(learning_rate=args.lr),
        "rmsprop": RMSprop(learning_rate=args.lr),
        "adam": Adam(learning_rate=args.lr),
        "sgd": SGD(learning_rate=args.lr)
    }
    model.compile(optimizer=optimizer_dict.get(args.learner.lower(), Adam(args.lr)),
                  loss='binary_crossentropy')

    # Init performance
    t1 = time()
    hits, ndcgs = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.mean(hits), np.mean(ndcgs)
    print(f'Init: HR = {hr:.4f}, NDCG = {ndcg:.4f} [{time()-t1:.1f} s]')

    # Train
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(args.epochs):
        t1 = time()
        user_input, item_input, labels = get_train_instances(train, args.num_neg, num_items)

        # Training
        history = model.fit(
            [user_input, item_input], labels,
            batch_size=args.batch_size, epochs=1, verbose=0, shuffle=True
        )
        t2 = time()

        # Evaluation
        if epoch % args.verbose == 0:
            hits, ndcgs = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.mean(hits), np.mean(ndcgs), history.history['loss'][0]
            print(f'Iteration {epoch} [{t2-t1:.1f} s]: HR = {hr:.4f}, NDCG = {ndcg:.4f}, loss = {loss:.4f}')
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file)

    print(f"End. Best Iteration {best_iter}: HR = {best_hr:.4f}, NDCG = {best_ndcg:.4f}.")
    if args.out > 0:
        print(f"The best MLP model is saved to {model_out_file}")
