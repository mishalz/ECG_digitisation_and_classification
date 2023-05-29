import pandas as pd
import numpy as np
import tensorflow as tf


relation_interpreter = tf.lite.Interpreter('./relatiomodel.tflite')
embedding_interpreter = tf.lite.Interpreter('./embeddingmodel.tflite')

relation_interpreter.allocate_tensors()
embedding_interpreter.allocate_tensors()


# get iut ad output of relation module
input_details1 = relation_interpreter.get_input_details()
output_details1 = relation_interpreter.get_output_details()

# get iut ad output of embedding module
input_details2 = embedding_interpreter.get_input_details()
output_details2 = embedding_interpreter.get_output_details()


test_data = './test.csv'
test = pd.read_csv(test_data)
test_x = test.iloc[:, :-1]
test_y = test.iloc[:, -1]  # labels of the data

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

test_x['label'] = test_y

for k in range(1, 10):
    mean_vecs = np.zeros((len(np.unique(test_y)), 128))  # suort set
    queries = np.empty((0, 128))  # query embeddig set
    query_labels = np.empty(0)  # query labels
    i = 0
    # take k samle from the query set
    for label in labels:

        samples = test_x[test_x['label'] == label].sample(n=k, replace=True)
        samples = samples.drop(samples.columns[-1], axis=1)

        samples = samples.values

        for support_sample in samples:
            support_sample = support_sample.reshape(1, -1, 1)  # o. of samle i batch, auomatically set, o. of chael

            support_sample = np.float32(support_sample)
            embedding_interpreter.allocate_tensors()
            embedding_interpreter.set_tensor(input_details2[0]['index'], support_sample)  # input to the model

            embedding_interpreter.invoke()

            samples_embedding = embedding_interpreter.get_tensor(output_details2[0]['index'])  # outut embeddigs

            class_feature = np.mean(samples_embedding, axis=0)
            mean_vecs[i] = class_feature

        query_samples = test_x[test_x['label'] == label].sample(n=5, replace=True)
        query_samples = query_samples.drop(query_samples.columns[-1], axis=1)
        query_samples = query_samples.values

        for sample in query_samples:
            sample = sample.reshape(1, -1, 1)

            sample = np.float32(sample)
            embedding_interpreter.allocate_tensors()
            embedding_interpreter.set_tensor(input_details2[0]['index'], sample)  # input to the model
            queries_embedding = embedding_interpreter.get_tensor(output_details2[0]['index'])  # outut embeddigs
            embedding_interpreter.invoke()
            queries = np.concatenate((queries, queries_embedding), axis=0)

        # Append the labels of the current class to the query labels
        query_labels = np.concatenate((query_labels, np.array([label] * 5)))

        i += 1
preds = []

for query in queries:

    similarity_scores = []

    # mult_query = np.resize(query, (13,128))

    query = np.float32(query)

    mean_vecs = np.float32(mean_vecs)

    for i in range(len(mean_vecs)):
        relation_interpreter.allocate_tensors()
        # the relatio module exects a iut of shae (13,128)
        relation_interpreter.set_tensor(input_details1[0]['index'], mean_vecs[i].reshape(1, 128))
        relation_interpreter.set_tensor(input_details1[1]['index'], query.reshape(1, 128))
        relation_interpreter.invoke()

        similarity_scores.append(relation_interpreter.get_tensor(output_details1[0]['index']))
    preds.append(np.argmax(similarity_scores))

#for q_label in range(len(query_labels)):
    #print("actual label: ", query_labels[q_label], "predicted label: ", preds[q_label])








