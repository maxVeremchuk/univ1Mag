from rouge import Rouge

if __name__ == "__main__":
    golden = []
    result = []
    with open("golden_summ.txt", "r") as f:
        golden = f.readlines()
    with open("result.txt", "r") as f:
        result = f.readlines()

    rouge = Rouge()
    scores = rouge.get_scores(golden, result, avg=True)
    print(scores)
