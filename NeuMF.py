import argparse
import numpy as np
from time import time
import multiprocessing as mp

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, Flatten, Multiply, Concatenate, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop

from Dataset import Dataset
from evaluate import evaluate_model
import GMF, MLP


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', type=str, default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', type=str, default='ml-1m',
                        help='Dataset name.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of GMF model.')
    parser.add_argument('--layers', type=str, default='[64,32,16,8]',
                        help='MLP layers. First layer size/2 = embedding size.')
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for GMF embeddings.')
    parser.add_argument('--reg_layers', type=str, default='[0,0,0,0]',
                        help='Regularization for each MLP layer.')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative samples per positive.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', type=str, default='adam',
                        help='Optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X epochs.')
    parser.add_argument('--out', type=int, default=1,
                        help='Save trained model.')
    parser.add_argument('--mf_pretrain', type=str, default='',
                        help='Pretrained GMF model file path.')
    parser.add_argument('--mlp_pretrain', type=str, default='',
                        help='Pretrained MLP model file path.')
    return parser.parse_args()


#################### Model ####################
def get_model(num_users: int, num_items: int,
              mf_dim=10, layers=None, reg_layers=None, reg_mf=0) -> Model:
    """
    Build the NeuMF model combining GMF and MLP.
    """
    if layers is None:
        layers = [64, 32, 16, 8]
    if reg_layers is None:
        reg_layers = [0] * len(layers)

    # Inputs
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # GMF part
    mf_user_latent = Flatten()(Embedding(
        input_dim=num_users,
        output_dim=mf_dim,
        name='mf_embedding_user',
        embeddings_initializer="random_normal",
        embeddings_regularizer=l2(reg_mf)
    )(user_input))

    mf_item_latent = Flatten()(Embedding(
        input_dim=num_items,
        output_dim=mf_dim,
        name='mf_embedding_item',
        embeddings_initializer="random_normal",
        embeddings_regularizer=l2(reg_mf)
    )(item_input))

    mf_vector = Multiply()([mf_user_latent, mf_item_latent])

    # MLP part
    mlp_embedding_dim = layers[0] // 2
    mlp_user_latent = Flatten()(Embedding(
        input_dim=num_users,
        output_dim=mlp_embedding_dim,
        name="mlp_embedding_user",
        embeddings_initializer="random_normal",
        embeddings_regularizer=l2(reg_layers[0])
    )(user_input))

    mlp_item_latent = Flatten()(Embedding(
        input_dim=num_items,
        output_dim=mlp_embedding_dim,
        name='mlp_embedding_item',
        embeddings_initializer="random_normal",
        embeddings_regularizer=l2(reg_layers[0])
    )(item_input))

    mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])

    for idx in range(1, len(layers)):
        mlp_vector = Dense(
            layers[idx], activation='relu',
            kernel_regularizer=l2(reg_layers[idx]),
            name=f"layer{idx}"
        )(mlp_vector)

    # Concatenate GMF + MLP
    predict_vector = Concatenate()([mf_vector, mlp_vector])

    # Final prediction
    prediction = Dense(1, activation='sigmoid', kernel_initializer="lecun_uniform", name="prediction")(predict_vector)

    return Model(inputs=[user_input, item_input], outputs=prediction)


#################### Pretrain Loader ####################
def load_pretrain_model(model, gmf_model, mlp_model, num_layers: int):
    """
    Load pretrained GMF and MLP into NeuMF.
    """
    # GMF embeddings
    model.get_layer('mf_embedding_user').set_weights(gmf_model.get_layer('user_embedding').get_weights())
    model.get_layer('mf_embedding_item').set_weights(gmf_model.get_layer('item_embedding').get_weights())

    # MLP embeddings
    model.get_layer('mlp_embedding_user').set_weights(mlp_model.get_layer('user_embedding').get_weights())
    model.get_layer('mlp_embedding_item').set_weights(mlp_model.get_layer('item_embedding').get_weights())

    # MLP layers
    for i in range(1, num_layers):
        model.get_layer(f'layer{i}').set_weights(mlp_model.get_layer(f'layer{i}').get_weights())

    # Prediction layer = average of GMF + MLP predictions
    gmf_pred = gmf_model.get_layer('prediction').get_weights()
    mlp_pred = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_pred[0], mlp_pred[0]), axis=0)
    new_bias = 0.5 * (gmf_pred[1] + mlp_pred[1])
    model.get_layer('prediction').set_weights([0.5 * new_weights, new_bias])

    return model


#################### Training Data ####################
def get_train_instances(train, num_negatives: int, num_items: int):
    user_input, item_input, labels = [], [], []
    for (u, i) in train.keys():
        # Positive
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # Negatives
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

    print("NeuMF arguments:", args)
    model_out_file = f'Pretrain/{args.dataset}_NeuMF_{args.num_factors}_{args.layers}_{int(time())}.h5'

    # Load data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.train_matrix, dataset.test_ratings, dataset.test_negatives
    num_users, num_items = train.shape
    print(f"Load data done [{time()-t1:.1f} s]. "
          f"#user={num_users}, #item={num_items}, #train={train.nnz}, #test={len(testRatings)}")

    # Build model
    model = get_model(num_users, num_items, args.num_factors, layers, reg_layers, args.reg_mf)

    # Compile
    optimizer_dict = {
        "adagrad": Adagrad(learning_rate=args.lr),
        "rmsprop": RMSprop(learning_rate=args.lr),
        "adam": Adam(learning_rate=args.lr),
        "sgd": SGD(learning_rate=args.lr)
    }
    model.compile(optimizer=optimizer_dict.get(args.learner.lower(), Adam(args.lr)),
                  loss='binary_crossentropy')

    # Load pretrained GMF + MLP
    if args.mf_pretrain and args.mlp_pretrain:
        gmf_model = GMF.get_model(num_users, num_items, args.num_factors)
        gmf_model.load_weights(args.mf_pretrain)
        mlp_model = MLP.get_model(num_users, num_items, layers, reg_layers)
        mlp_model.load_weights(args.mlp_pretrain)
        model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
        print(f"Loaded pretrained GMF ({args.mf_pretrain}) and MLP ({args.mlp_pretrain})")

    # Init performance
    hits, ndcgs = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.mean(hits), np.mean(ndcgs)
    print(f'Init: HR = {hr:.4f}, NDCG = {ndcg:.4f}')
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    if args.out > 0:
        model.save_weights(model_out_file)

    # Training
    for epoch in range(args.epochs):
        t1 = time()
        user_input, item_input, labels = get_train_instances(train, args.num_neg, num_items)

        # Train
        hist = model.fit([user_input, item_input], labels,
                         batch_size=args.batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()

        # Eval
        if epoch % args.verbose == 0:
            hits, ndcgs = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.mean(hits), np.mean(ndcgs), hist.history['loss'][0]
            print(f'Iteration {epoch} [{t2-t1:.1f} s]: HR = {hr:.4f}, NDCG = {ndcg:.4f}, loss = {loss:.4f}')
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file)

    print(f"End. Best Iteration {best_iter}: HR = {best_hr:.4f}, NDCG = {best_ndcg:.4f}.")
    if args.out > 0:
        print(f"The best NeuMF model is saved to {model_out_file}")
