from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import torch
np.random.seed(42)
torch.manual_seed(42)


def load_tokenizer_and_model(model_name_or_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    return tokenizer, GPT2LMHeadModel.from_pretrained(
        model_name_or_path,
        pad_token_id=tokenizer.eos_token_id
    )


def generate(
    model,
    tokenizer,
    text,
    do_sample=True,
    max_length=50,
    repetition_penalty=5.0,
    top_k=5,
    top_p=0.95,
    temperature=1,
    num_beams=5,
    no_repeat_ngram_size=2,
    num_return_sequences=5,
    early_stopping=True
):

    encoded_text = tokenizer.encode(
        text,
        add_special_tokens=False,
        return_tensors="pt"
    )

    out = model.generate(
        input_ids=encoded_text,
        max_length=max_length + len(encoded_text[0]),
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_return_sequences=num_return_sequences,
        early_stopping=early_stopping
    )

    return list(map(tokenizer.decode, out))


t, m = load_tokenizer_and_model("sberbank-ai/rugpt3large_based_on_gpt2")
generated = generate(
    model=m,
    tokenizer=t,
    text="Я шел по улице, ноги мои промокли от дождя. Я думал о том дне и не мог понять: почему мне так плохо? Почему я чувствую себя таким одиноким в этом городе? Я прожил здесь 10 лет и ни разу с тех пор ничего подобного со мной еще никогда небыло… Может быть потому что у меня нет семьи? В тот день умер мой пёс Тарковский. У него был рак печени на последней стадии – это было видно невооруженным глазом из-за того как он мучился последние дни своей жизни. Я поздно узнал об этом и винил себя в смерти собаки. Надо было внимательнее следить, - подумал я про себя. Но теперь все изменилось! Теперь моя жизнь наполнена смыслом... Ведь сегодня я уезжаю",
    max_length=30,
    num_beams=10,
    top_k=0,
    top_p=0.92,
    num_return_sequences=1
)

# print(generated)
for i, output in enumerate(generated):
    print(output)
