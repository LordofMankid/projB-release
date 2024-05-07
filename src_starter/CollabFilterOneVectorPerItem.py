'''
CollabFilterOneVectorPerItem.py

Defines class: `CollabFilterOneVectorPerItem`

Scroll down to __main__ to see a usage example.
'''

# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd.numpy as ag_np

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets

# Some packages you might need (uncomment as necessary)
import pandas as pd
import matplotlib.pyplot as plt

# No other imports specific to ML (e.g. scikit) needed!

class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    '''

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        '''
        random_state = self.random_state # inherited RandomState object

        # TODO fix the lines below to have right dimensionality & values
        # TIP: use self.n_factors to access number of hidden dimensions
        self.param_dict = dict(
            mu=ag_np.ones(1),
            b_per_user=ag_np.ones(n_users), # FIX dimensionality
            c_per_item=ag_np.ones(n_items), # FIX dimensionality
            U=0.01 * random_state.randn(n_users, self.n_factors), # FIX dimensionality
            V=0.01 * random_state.randn(n_items, self.n_factors), # FIX dimensionality
        )


    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''
         # Extract user and item factors based on provided IDs
        user_factors = U[user_id_N]  # shape (N, n_factors)
        item_factors = V[item_id_N]  # shape (N, n_factors)

        # Compute the dot product for each user-item pair
        # (N, n_factors) * (N, n_factors) -> (N, ) via summing over the factors
        dot_product = ag_np.sum(user_factors * item_factors, axis=1)

        # Compute predictions by adding bias terms and the dot product results
        yhat_N = mu + b_per_user[user_id_N] + c_per_item[item_id_N] + dot_product

        return yhat_N


    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        '''
        # TODO compute loss
        # TIP: use self.alpha to access regularization strength
        y_N = data_tuple[2]
        
        yhat_N = self.predict(data_tuple[0], data_tuple[1], **param_dict)
        reg_term = self.alpha * (ag_np.sum(param_dict['U'][data_tuple[0]] ** 2) + ag_np.sum(param_dict['V'][data_tuple[1]] ** 2))

        loss_total = reg_term + ag_np.sum((y_N - yhat_N) ** 2)
             
        return loss_total    


if __name__ == '__main__':

    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()
    # Create the model and initialize its parameters
    # to have right scale as the dataset (right num users and items)
    
    K = 50
    alpha = [0.9, 1, 1.1, 1.2]
    fig, axes = plt.subplots(1, 1, figsize=(12, 3.5), sharey=True)
    
    models = []

    # import movies
        # for i in range(0, 3):
    #     model = CollabFilterOneVectorPerItem(
    #         n_epochs=40, batch_size=32, step_size=0.2,
    #         n_factors=10, alpha=alpha[i])
    #     model.init_parameter_dict(n_users, n_items, train_tuple)

    #     # Fit the model with SGD
    #     model.fit(train_tuple, valid_tuple)
    #     models.append[model]
    #     print(model.evaluate_perf_metrics(train_tuple[0], train_tuple[1], train_tuple[2]))


    #     # Plotting the loss trace
    #     axes[i].plot(model.trace_epoch, model.trace_train_mae, label='Training MAE', color='blue')
    #     axes[i].plot(model.trace_epoch, model.trace_valid_mae, label='Validation MAE', color='green')
    #     axes[i].set_title('MAE Trace')
    #     axes[i].set_xlabel('Epoch')
    #     axes[i].set_ylabel('Mean Absolute Error')

    movies_df = pd.read_csv('../data_movie_lens_100k/select_movies.csv')

    print(movies_df.head())
    
    import dill as pickle
    loaded_model = None

    try:
        with open('saved_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Saved model not found. Training a new model.")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}. Training a new model.")

    # If the model is not loaded, initialize and train a new one
    if not loaded_model:
        model = CollabFilterOneVectorPerItem(
            n_epochs=10, batch_size=32, step_size=0.2,
            n_factors=2, alpha=1)
        model.init_parameter_dict(n_users, n_items, train_tuple)
        model.fit(train_tuple, valid_tuple)
        
        with open('saved_model.pkl', 'wb') as file:
            pickle.dump(model, file)
        print("Model trained and saved.")
    else:
        model = loaded_model


    U = model.param_dict['U']  # user embeddings
    V = model.param_dict['V']  # item embeddings
    print(V.shape)

    selected_movie_ids = movies_df['item_id'].values  # get numpy array
    
    selected_years = movies_df['release_year'].values
    
    print(selected_movie_ids)
    selected_movie_embeddings = model.param_dict['V'][selected_movie_ids]
    
    X_axis = selected_movie_embeddings[:, 0]
    Y_axis = selected_movie_embeddings[:, 1]

    # Determine the limits for the plot to center at (0,0)
    x_min, x_max = X_axis.min(), X_axis.max()
    y_min, y_max = Y_axis.min(), Y_axis.max()

    # Calculate maximum range to ensure (0,0) is centered
    x_limit = max(abs(x_min), abs(x_max))
    y_limit = max(abs(y_min), abs(y_max))

    # Plot configuration
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_axis, Y_axis, s=100, alpha=0.5)

    # Adding cross axes at x=0 and y=0
    plt.axhline(0, color='black', linewidth=0.5)  # Horizontal line at y=0
    plt.axvline(0, color='black', linewidth=0.5)  # Vertical line at x=0

    # Set x and y limits to center (0,0)
    plt.xlim(-x_limit, x_limit)
    plt.ylim(-y_limit, y_limit)

    plt.xticks([])
    plt.yticks([])

    # Label each point with movie title
    for i, text in enumerate(movies_df['title']):
        plt.annotate(text, (X_axis[i], Y_axis[i]))

    plt.xlabel('Factor 1')
    plt.ylabel('Factor 2')
    plt.title('Selected Movies as Embedding Vectors')
    plt.show()