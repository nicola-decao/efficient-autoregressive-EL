import os

import jsonlines
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.data.mulrel_nel_dataset import CoNLLDataset


def convert_aida(dataset):

    new_dataset = {}
    for k, d in tqdm(dataset.items()):

        doc_str = ""
        offsets = [[]]
        for s in d[0]["conll_doc"]["sentences"]:
            for w in s:
                offsets[-1].append((len(doc_str), len(doc_str) + len(w)))
                doc_str += w
                doc_str += " "
            offsets.append([])

        doc = tokenizer(
            doc_str[:-1],
            return_offsets_mapping=True,
        )

        new_dataset[k] = {
            "id": k,
            "input": doc_str[:-1],
            "anchors": [],
            "candidates": [],
        }
        for e in d[0]["conll_doc"]["mentions"]:
            find_start_token = False
            for start_token, t in enumerate(doc["offset_mapping"]):
                if t[0] == offsets[e["sent_id"]][e["start"]][0]:
                    find_start_token = True
                    break

            find_end_token = False
            for end_token, t in enumerate(doc["offset_mapping"]):
                if t[1] == offsets[e["sent_id"]][e["end"] - 1][1]:
                    find_end_token = True
                    break

            assert find_start_token and find_end_token

            new_dataset[k]["anchors"].append(
                [
                    start_token,
                    end_token,
                    e["wikilink"]
                    .replace("http://en.wikipedia.org/wiki/", "")
                    .replace("_", " "),
                ]
            )

        with open(
            f'/home/ndecao/PPRforNED/AIDA_candidates/{k.split(" ")[0].replace("testa", "").replace("testb", "")}'
        ) as f:
            l = [e.strip().split("\t") for e in f]

        status = "e"
        for e in l:
            if e[0] == "ENTITY":
                status == "e"
                if e[-1] != "url:NIL":
                    new_dataset[k]["anchors"][len(new_dataset[k]["candidates"])][
                        -1
                    ] = e[-1][33:].replace("_", " ")
                    new_dataset[k]["candidates"].append(set())
                    status = "c"
            elif e[0] == "CANDIDATE" and status == "c":
                new_dataset[k]["candidates"][-1].add(e[5][33:].replace("_", " "))

        new_dataset[k]["candidates"] = [list(e) for e in new_dataset[k]["candidates"]]

        if len(new_dataset[k]["candidates"]) != len(new_dataset[k]["anchors"]):
            print(k)
            new_dataset[k]["anchors"] = new_dataset[k]["anchors"][
                : len(new_dataset[k]["candidates"])
            ]
            assert len(new_dataset[k]["candidates"]) == len(new_dataset[k]["anchors"])

    return new_dataset


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

    path = "./data/mulrel_nel_dataset/generated/test_train_data/"
    conll_path = "./data/mulrel_nel_dataset/basic_data/test_datasets/"
    person_path = "./data/mulrel_nel_dataset/basic_data/p_e_m_data/persons.txt"

    dataset = CoNLLDataset(path, person_path, conll_path)

    for split_name, split_dataset in zip(
        ("train", "val", "test"), (dataset.train, dataset.testA, dataset.testB)
    ):
        split_dataset = convert_aida(split_dataset)
        with jsonlines.open(f"./data/aida_{split_name}_dataset.jsonl", "w") as f:
            f.write_all(list(split_dataset.values()))

    with open(
        "/home/ndecao/end2end_neural_el/data/entities/entities_universe.txt"
    ) as f:
        entities_universe = [e.strip().split("\t")[1].replace("_", " ") for e in f]

    with open("../data/entities.json", "w") as f:
        json.dump(entities_universe, f)

    with open("../data/aida_means.tsv", newline="") as csvfile:
        mentions = [row[0].strip() for row in csv.reader(csvfile, delimiter="\t")]

    with open("../data/mentions.json", "w") as f:
        json.dump(mentions, f)
