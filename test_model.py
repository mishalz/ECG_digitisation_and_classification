import load_model
import pandas as pd
import numpy as np


class test_model:
    def compute(self,file_path):

        mean = load_model.mean_vecs
        embedding_interpreter = load_model.embedding_interpreter
        relation_interpreter = load_model.relation_interpreter

        input_details1 = load_model.input_details1
        output_details1 = load_model.output_details1

        input_details2 = load_model.input_details2
        output_details2 = load_model.output_details2

        test_file = file_path
        test1 = pd.read_csv(test_file)

        test_x1 = test1.iloc[:, 4:238]
        test_y1 = test1.label

        test_x1['label'] = test_y1

        test_queries = np.empty((0, 128))  # query embedding set
        test_query_labels = np.empty(0)  # query labels

        for label in np.sort(np.unique(test_y1)):

            test_query_samples = test_x1[test_x1['label'] == label].sample(n=1, replace=True)
            test_query_samples = test_query_samples.drop(test_query_samples.columns[-1], axis=1)
            test_query_samples = test_query_samples.values

            for sample in test_query_samples:
                sample = sample.reshape(1, -1, 1)

                sample = np.float32(sample)
                embedding_interpreter.allocate_tensors()
                embedding_interpreter.set_tensor(input_details2[0]['index'], sample)  # input to the model
                queries_embedding = embedding_interpreter.get_tensor(output_details2[0]['index'])  # outut embeddigs
                embedding_interpreter.invoke()
                test_queries = np.concatenate((test_queries, queries_embedding), axis=0)

                # Append the labels of the current class to the query labels
            test_query_labels = np.concatenate((test_query_labels, np.array([label])))

        test_preds = []
        for query in test_queries:

            test_similarity_scores = []

            query = np.float32(query)

            for i in range(len(mean)):
                relation_interpreter.allocate_tensors()
                # the relatio module exects a iut of shae (13,128)
                relation_interpreter.set_tensor(input_details1[0]['index'], mean[i].reshape(1, 128))
                relation_interpreter.set_tensor(input_details1[1]['index'], query.reshape(1, 128))
                relation_interpreter.invoke()

                test_similarity_scores.append(relation_interpreter.get_tensor(output_details1[0]['index']))
            test_preds.append(np.argmax(test_similarity_scores))

        #for q_label in range(len(test_query_labels)):
            #print("actual label: ", test_query_labels[q_label], "predicted label: ", test_preds[q_label])

        return test_query_labels, test_preds
