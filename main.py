from LanguageModel import LanguageModel
import requests

if __name__ == "__main__":
    model = LanguageModel(4, missed_value=0.9)
    response = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
    model.build_model(text=response.text)
    print(model.calculate_proba("Our business is not unknown to the king"))
    print(model.calculate_proba("Our business is not unknown to the boss"))
