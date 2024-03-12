from openai import OpenAI

boolq_examples = [{'question': 'can an ethernet cable be used for poe', 'answer': True}, {'question': 'is argentina the largest country in south america', 'answer': False}, {'question': 'is the gulf of mexico part of an ocean', 'answer': True}, {'question': 'is the sierra nevada part of the rocky mountains', 'answer': False}, {'question': 'does cleveland state university have a football team', 'answer': True}, {'question': 'is first round of hockey best of 5', 'answer': False}, {'question': "is a mare's leg considered a pistol", 'answer': True}, {'question': 'is dazed and confused based on a true story', 'answer': False}, {'question': 'is rise of the tomb raider connected to tomb raider', 'answer': True}, {'question': 'is the poseidon adventure movie based on a true story', 'answer': False}, {'question': "do they find out that katherine is in elena's body", 'answer': True}, {'question': 'is bahrain the smallest country in the world', 'answer': False}, {'question': 'can an analog signal be converted to digital', 'answer': True}, {'question': 'are sodium chloride and saline the same thing', 'answer': False}, {'question': 'has wayne rooney ever won the champions league', 'answer': True}, {'question': 'is there a season 2 of wits academy', 'answer': False}, {'question': 'is the new god of war ps4 exclusive', 'answer': True}, {'question': 'is the wall movie based on true story', 'answer': False}, {'question': 'can a car go the speed of sound', 'answer': True}, {'question': 'do all countries in eu use the euro', 'answer': False}, {'question': 'does a white blood cell have a nucleus', 'answer': True}, {'question': 'do ross and rachel end up getting married', 'answer': False}, {'question': 'can you have a delayed reaction to grief', 'answer': True}, {'question': 'is dark places based on a true story', 'answer': False}, {'question': 'is harry potter world at islands of adventure', 'answer': True}, {'question': 'does the sig sauer p226 have a safety', 'answer': False}, {'question': "is the handmaid's tale a hulu original", 'answer': True}, {'question': 'is the movie morning glory based on a true story', 'answer': False}, {'question': 'does each australian state have its own constitution', 'answer': True}, {'question': 'will there be any more seasons of the paradise', 'answer': False}, {'question': 'was the bill of rights added to the constitution', 'answer': True}]

def evaluate_model():
    client = OpenAI()
    total = len(boolq_examples)
    correct = 0 
    for qa in boolq_examples:
        answer = qa['answer']
        prompt = f"{qa['question']}\n"
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=0,
            max_tokens=2,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        response = response.choices[0].text.strip()
        model_ans = True if response == 'Yes' else False
        print(model_ans, answer)
        if model_ans == answer:
            correct += 1

    return correct / total

if __name__ == "__main__":
    accuracy = evaluate_model()
    print(f"Evaluation Accuracy: {accuracy}")
