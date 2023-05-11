from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tensorflow import keras
from keras.models import load_model
import numpy as np


class test_model():
    model = load_model('trainedmodel.h5')

    def sample_k_shot_task_SCNN(X, y, k, q, seed_state_support, seed_state_query):
        """
            Sample a FSL k-shot task for the SCNN model
            Parameters
            ----------
            X : pandas.DataFrame
                Input data.
            y : pandas.Series
                Input labels.
            k : int
                Number of samples per class.
            q : int
                Number of queries per class.
            seed_state_support : int
                Seed of support set.
            seed_state_query : int
                Seed of query set.

        """
        # Initialize the mean vectors, query set and query labels
        mean_vecs = np.zeros((len(np.unique(y)), 128))
        queries = np.empty((0, 128))
        query_labels = np.empty(0)
        i = 0
        # Iterate over the unique labels
        for label in np.sort(np.unique(y)):
            # Sample k instances of the current label for the support set, and post-pad with zeros
            samples = X[y == label].sample(
                n=k, replace=True, random_state=seed_state_support)
            # samples_pad = tf.keras.preprocessing.sequence.pad_sequences(samples.values , maxlen = 234,
            # dtype = 'float64', padding = 'post')

            # Obtain embeddings of the padded samples of the support set - f(X)
            samples_embedding = model.embedding_module.predict(samples)
            # Average the embeddings of the padded samples to get the mean vector of the class mu
            class_feature = np.mean(samples_embedding, axis=0)
            # Append the mean vector of the current class to the mean vectors matrix
            mean_vecs[i] = class_feature

            # Sample q instances of the current label for the query set, and post-pad with zeros
            query_samples = X[y == label].sample(
                n=q, replace=True, random_state=seed_state_query)
            # query_pad = tf.keras.preprocessing.sequence.pad_sequences(query_samples.values , maxlen = 234,
            # dtype = 'float64', padding = 'post')

            # Obtain embeddings of the padded samples of the query set - Q
            queries_embedding = model.embedding_module.predict(query_samples)
            # Append the embeddings of the query set to queries
            queries = np.concatenate((queries, queries_embedding), axis=0)

            # Append the labels of the current class to the query labels
            query_labels = np.concatenate(
                (query_labels, np.array([label] * q)))

            i += 1

        return mean_vecs, queries, query_labels

    def get_metrics_of_task_SCNN(mean_vecs, queries, query_labels):
        """
            Get evaluation metrics of one task of SCNN from support set and query set
            Parameters
            ----------
            mean_vecs: numpy.ndarray
                Mean vectors of each class of the support set
            queries: numpy.ndarray
                The query set
            query_labels: numpy.ndarray
                Labels of the queries
            Returns
            -------
            acc : float
                Accuracy of task
            prec : float
                Macro-averaged precision of task
            rec : float
                Macro-averaged recall of task
            f1 : float
                Macro-averaged F1-score of task
        """
        print("in here")
        preds = []
        for query in queries:
            mult_query = np.resize(query, (13, 128))

            # Get similarity score g( mu, Q ) for each mean vector mu and and the query embedding Q
            similarity_scores = model.relation_module.predict(
                [mean_vecs, mult_query])
            # Final prediction of the query is the index of the class with the highest similarity score
            preds.append(np.argmax(similarity_scores))

        for q_label in range(len(query_labels)):
            print("actual label: ",
                  query_labels[q_label], "predicted label: ", preds[q_label])

        acc = accuracy_score(query_labels, preds)
        recall = recall_score(query_labels, preds, average='macro')
        precision = precision_score(query_labels, preds, average='macro')
        f1 = f1_score(query_labels, preds, average='macro')

        return acc, recall, precision, f1

    def get_metrics_k_shot_SCNN(X, y, k, q, n_tasks):
        """
            Get evaluation metrics of k-shot tasks of the SCNN from a dataset
            Parameters
            ----------
            X : pandas.DataFrame
                Input data.
            y : pandas.Series
                Input labels.
            k : int
                Number of samples per class of support set.
            q : int
                Number of samples per class of query set.
            n_tasks : int
                Number of tasks.
            Returns
            -------
            accuracies : numpy.ndarray
                List of accuracies of each k-shot task of size (n_tasks, )
            recalls : numpy.ndarray
                List of macro-averaged recalls of each k-shot task of size (n_tasks, )
            precisions : numpy.ndarray
                List of macro-averaged precisions of each k-shot task of size (n_tasks, )
            f1s : numpy.ndarray
                List of macro-averaged F1-scores of each k-shot task of size (n_tasks, )
        """
        accuracies = []
        recalls = []
        precisions = []
        f1s = []

        for i in tqdm(range(n_tasks)):
            # Sample k-shot task
            mean_vecs, queries, query_labels = sample_k_shot_task_SCNN(X, y, k, q, seed_state_support=i,
                                                                       seed_state_query=n_tasks + i)
            print("here2")
            # Get the evaluation metrics of the k-shot task
            acc, recall, precision, f1 = get_metrics_of_task_SCNN(
                mean_vecs, queries, query_labels)

            accuracies.append(acc)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
        return accuracies, recalls, precisions, f1s

    SCNN_accs = []
    SCNN_recalls = []
    SCNN_precisions = []
    SCNN_F1s = []

    def print_metrics_SCNN(X_test, y_test, k_min, k_max, num_tasks, q):
        """
            Print evaluation metrics of k-shot tasks of SCNN from a dataset
            Parameters
            ----------
            X_test : pandas.DataFrame
                Input data.
            y_test : pandas.Series
                Input labels.
            k_min : int
                Minimum number of samples per class of support set.
            k_max : int
                Maximum number of samples per class of support set.
            num_tasks : int
                Number of SCNN tasks to run.
            q : int
                Number of samples per class of query set.
        """
        for i in range(k_min, k_max + 1):
            accuracies, recalls, precisions, f1s = get_metrics_k_shot_SCNN(
                X_test, y_test, k=i, q=q, n_tasks=num_tasks)
            print('For %d shot, Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f ' % (
                i, np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f1s)))

            SCNN_accs.append(np.mean(accuracies))
            SCNN_recalls.append(np.mean(recalls))
            SCNN_precisions.append(np.mean(precisions))
            SCNN_F1s.append(np.mean(f1s))

    test_label = test_y.values
    print_metrics_SCNN(test_df, test_label, 1, 10, 10, 5)
