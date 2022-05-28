from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("flexudy/t5-base-multi-sentence-doctor")

model = AutoModelWithLMHead.from_pretrained("flexudy/t5-base-multi-sentence-doctor")


def correct_sentence(repair_sentence, context):
    """
        Create complete sentence using input and context
    :param repair_sentence: extracted strip of sentence
    :param context: whole input sentence
    :return:
    """
    
    input_text = 'repair_sentence: ' + repair_sentence + ' ' + 'context:{' + context + '} </s>'
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    outputs = model.generate(input_ids, max_length=100, num_beams=1)

    sentence = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return sentence
