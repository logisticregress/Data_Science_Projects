from langchain import OpenAI, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI


def create_index(directory_path):
    max_input_len = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_len, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo', max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    print("the error is here")
    print(documents)

    if not documents:
        documents = ["I don't know anything yet"]
    
    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    print("index", index)

    # Save index to disk
    try:
        index.save_to_disk('index.json')
        print("Index saved to disk")
    except Exception as e:
        print(f"Error saving index to disk: {e}")
    
    return index
