from transformers import pipeline
generator_gpt2 = pipeline('text-generation', model = 'gpt2')
text2text_generator = pipeline("text2text-generation", model = "t5-small")

my_file = open('./TURK_Original.txt', 'r')
data = my_file.read()
splitting = data.split("\n")

print(splitting[0])

for i in range (0, len(splitting)-1):
  response = generator_gpt2(f"Simplify the following text: {splitting[i]}", max_length = 30, num_return_sequences=3)[0]['generated_text']
  with open('./TURK_GPT2.txt', 'a') as f:
    f.write(response.rstrip('\r\n'))
    f.write('\n')
  

f.close()