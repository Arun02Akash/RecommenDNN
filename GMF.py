import numpy as np
import argparse
import multiprocessing as mp
from time import time

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, Flatten, Multiply, Dense
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2

from Dataset import Dataset
from evaluate import evaluate_model


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--path', type=str, default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', type=str, default='ml-1m',
                        help='Dataset name.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--regs', type=str, default='[0,0]',
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative samples per positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', type=str, default='adam',
                        help='Optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance every X epochs.')
    parser.add_argument('--out', type=int, default=1,
                        help='Save the trained model.')
    return parser.parse_args()


#################### Model ####################
def get_model(num_users: int, num_items: int, latent_dim: int, regs=[0, 0]) -> Model:
    """
    Build the GMF model.
    """
    # Inputs
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # Embeddings
    user_embedding = Embedding(
        input_dim=num_users, output_dim=latent_dim,
        name='user_embedding',
        embeddings_initializer="random_normal",
        embeddings_regularizer=l2(regs[0])
    )(user_input)

    item_embedding = Embedding(
        input_dim=num_items, output_dim=latent_dim,
        name='item_embedding',
        embeddings_initializer="random_normal",
        embeddings_regularizer=l2(regs[1])
    )(item_input)

    # Flatten
    user_latent = Flatten()(user_embedding)
    item_latent = Flatten()(item_embedding)

    # Element-wise product
    predict_vector = Multiply()([user_latent, item_latent])

    # Output
    prediction = Dense(1, activation='sigmoid', name='prediction')(predict_vector)

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
    regs = eval(args.regs)

    topK = 10
    evaluation_threads = mp.cpu_count()

    print("GMF arguments:", args)
    model_out_file = f'Pretrain/{args.dataset}_GMF_{args.num_factors}_{int(time())}.h5'

    # Load data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.train_matrix, dataset.test_ratings, dataset.test_negatives
    num_users, num_items = train.shape
    print(f"Load data done [{time()-t1:.1f} s]. "
          f"#user={num_users}, #item={num_items}, #train={train.nnz}, #test={len(testRatings)}")

    # Build model
    model = get_model(num_users, num_items, args.num_factors, regs)

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
    print(f'Init: HR = {hr:.4f}, NDCG = {ndcg:.4f}\t [{time()-t1:.1f} s]')

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

        # Evaluation
        if epoch % args.verbose == 0:
            hits, ndcgs = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.mean(hits), np.mean(ndcgs), history.history['loss'][0]
            print(f'Iteration {epoch} [{time()-t1:.1f} s]: '
                  f'HR = {hr:.4f}, NDCG = {ndcg:.4f}, loss = {loss:.4f}')

            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file)

    print(f"End. Best Iteration {best_iter}: HR = {best_hr:.4f}, NDCG = {best_ndcg:.4f}.")
    if args.out > 0:
        print(f"The best GMF model is saved to {model_out_file}")
