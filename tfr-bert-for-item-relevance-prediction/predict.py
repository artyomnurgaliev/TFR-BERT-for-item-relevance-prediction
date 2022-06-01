import numpy as np
import pandas as pd
from tensorflow_serving.apis import input_pb2
import tqdm
from tfrecords import category_example, item_example, process_df
import json
import numpy
import requests
import base64
from official.nlp.bert import tokenization
import os
from config import bert_path

data_path = "./data/eval.csv"
content_df = pd.read_csv(data_path)
# if you want to see only titles
all_content = content_df["title"].values.tolist()


def create_records(df, tokenizer):
    all_categories = list(set(df.category_id.values.tolist()))
    all_records = []

    for category in tqdm.tqdm(all_categories):
        category_df = df.loc[df["category_id"] == category]
        category_name = category_df["category"].values.tolist()[0]
        CONTEXT = category_example(category_name)

        EXAMPLES = []
        item_titles, item_descriptions, item_brands, item_prices, ranks = process_df(category_df)
        for j in range(len(item_titles)):
            EXAMPLES.append(item_example(tokenizer, category_name, item_titles[j], item_descriptions[j],
                                         item_brands[j], item_prices[j], ranks[j]))

        ELWC = input_pb2.ExampleListWithContext()
        ELWC.context.CopyFrom(CONTEXT)
        for example in EXAMPLES:
            example_features = ELWC.examples.add()
            example_features.CopyFrom(example)

        all_records.append(ELWC)
    return all_records


def process_predictions(preds, top_k):
    top_k_idx = np.array(preds).argsort()[-top_k:][::-1]
    top_k_content = [str(all_content[idx]) for idx in top_k_idx]
    return top_k_content


if __name__ == "__main__":
    top_k = 10
    df = pd.read_csv(data_path)

    tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(bert_path, "vocab.txt"), do_lower_case=True)
    recs = create_records(df, tokenizer)

    recs = [example.SerializeToString() for example in recs]
    recs = [base64.b64encode(example) for example in recs]
    recs = [example.decode('utf-8') for example in recs]

    recommendations = []
    for idx, rec in enumerate(recs):
        data = json.dumps({"signature_name": "serving_default",
                           "instances": [{"b64": rec}]})
        headers = {"content-type": "application/json"}
        json_response = requests.post('http://localhost:8501/v1/models/tfrbert:predict',
                                      data=data, headers=headers)
        predictions = numpy.array(json.loads(json_response.text)["predictions"])[0]
        top_predictions = process_predictions(predictions, top_k)
        recommendations.append(",".join(top_predictions))

    recs_df = pd.DataFrame({"categories": list(set(df["category"].values.tolist())),
                            "recommendations": recommendations})
    recs_df.to_csv("sample_recs.csv", index=False)
