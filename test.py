from transformers import GPT2LMHeadModel, GPT2Tokenizer


model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

def generator (text):
  encoded_text = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
  encoded_text = encoded_text.to('cpu')
  out = model.generate(encoded_text, 10 + len(encoded_text[0]))
  generated_text = list(map(tokenizer.decode, out))[0]
  print('Seed:')
  print(text)
  print('Result:')
  print(generated_text)
  #generator(generated_text)

generator('Я проснулся в лесу,')