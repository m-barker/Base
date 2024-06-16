import argparse
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from lasr_vector_databases_faiss import (
    load_model,
    parse_txt_file,
    get_sentence_embeddings,
)

random.seed(42)


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--txt_path",
        type=str,
        help="Path to the txt file containing the sentences to embed",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Name of the model to load",
    )
    return vars(parser.parse_args())


def main():
    args = parse_args()

    model = load_model(args["model_name"])
    sentences = parse_txt_file(args["txt_path"])
    verb_dict = {
        "take": ["take", "get", "grasp", "fetch"],
        "place": ["put", "place"],
        "deliver": ["bring", "give", "deliver"],
        "bring": ["bring", "give"],
        "go": ["go", "navigate"],
        "find": ["find", "locate", "look for"],
        "talk": ["tell", "say"],
        "answer": ["answer"],
        "meet": ["meet"],
        "tell": ["tell"],
        "greet": ["greet", "salute", "say hello to", "introduce yourself to"],
        "count": ["tell me how many"],
        "follow": ["follow"],
        "guide": ["guide", "escort", "take", "lead"],
    }

    # shuffle sentences
    random.shuffle(sentences)

    sentences = sentences[:50000]
    subset = []
    verbs = []
    for verb in verb_dict:
        for syn in verb_dict[verb]:
            for sentence in sentences:
                if sentence in subset:
                    continue
                if syn in sentence:
                    subset.append(sentence)
                    verbs.append(verb)

    embeddings = get_sentence_embeddings(subset, model)

    compressed = TSNE(n_components=2).fit_transform(embeddings)

    df = pd.DataFrame(compressed, columns=["x", "y"])
    df["verb"] = verbs

    # scatter coloured by verb
    fig, ax = plt.subplots()
    for verb in verb_dict:
        df_verb = df[df["verb"] == verb]
        ax.scatter(df_verb["x"], df_verb["y"], label=verb)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
