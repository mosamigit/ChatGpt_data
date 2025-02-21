#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import tiktoken
import openai
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
import numpy as np
import sys
import time
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from colorama import Fore, Back, Style
MAX_CONTEXT_QUESTIONS = 2
####This code will remember past conversation.Currenlty we have saved past 3 responses because of rate limit issue
################################################################################
### Step 1
################################################################################                                                                                                                                                                                                   


#load_dotenv(Path(r"C:\Users\norin.saiyed\web-crawl-q-and-a\.env"))
load_dotenv()
# api_key = os.environ["API_KEY"]
# openai.api_key = api_key
openai.api_key = 'sk-kZAcnb4xH3LbPHI4n2sWT3BlbkFJvuC8rNr'
def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question_2(
    df,
    model="gpt-3.5-turbo",
    #text-davinci-003
    question="",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=1800,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        
        print("\n\n")

    try:
        #print("Context:\n" + context)
        # Create a completions using the question and context
        #prompt=f"I want you to act as Information service provider, which is specifically design for Inxeption.Your task is to give answers and provide usefull in depth information.Answer the question based on the context below, try to understand the context,user can ask questions based on synonyms or by twiking sentences.if the question can't be answered based on the context, say \"I don't know\".Do not mention anything about context in the response.Do not ask user to check context,provide URL in the response only.\n\nContext: {context}\n\nQuestion: {question}",
        prompt= f"Question: {question}\nAnswer:",
        #prompt= f"Question: {question}\nAnswer:",
        #instructions = f"""I want you to act as Information service provider, which is specifically design for Inxeption.Your task is to give answers and provide usefull in depth information.
        #                    Answer the question based on the context below, try to understand the context,user can ask questions based on synonyms or by twiking sentences.
        #                    If you are unable to provide an answer based on the context, please respond with the phrase \" I can't help with that.\".
        #                    Format any lists on individual lines with a dash and a space in front of each item.If question is not based on the context then,You must ask questions before answering to understand better what I am seeking.
        #                    seperate your lines with line break.If it contains a sequence of instructions ,rewrite those instruction in the following format: step1-...
        #                    Answers are intended for consumers so should include relevent product URL.Use HTML format to show URL
        #                    \n\nContext: {context}\n\n"""
        instructions = f"""Create a custom solar panel recommendation chatbot for Inxeption, designed to offer alternative product suggestions based on weight, height, and wattage inputs.

- The chatbot should understand user queries, even when using synonyms or rephrasing sentences.
look for different brands but provide accurate suggestion related to weight and wattage.
- If a specific inverter is mentioned and available in the trained context, offer accurate details, specifications, pricing, and the correct product URL.
- If the user doesn't mention a particular panel or inverter, provide general information about solar panels and inverters, covering benefits, installation process, maintenance, efficiency, and available brands.
- Ensure that product URLs generated for solar panels and inverters are valid and lead to the correct product pages in Inxeption's marketplace.
- Implement effective error handling to request clarification when questions are unclear or do not match the context.
- Utilize the power of the GPT-3.5 Turbo model to generate natural language responses based on context and user inputs.
- Continuously update the database with new solar panel and inverter products and their details for ongoing learning and improved user interactions.
- Do not include alternative products suggeston until user is asking to do so,If user is asking to provide alternative products based on weight or wattage then provide only two product suggestions.
- complete responce related to particular question would be,
-Question:Tiger Pro 72HCBDVP 530550 Watt Bifacial Module with 
Dual Glass 550W
-Answer:Sure! Here are the details of the Tiger Pro 72HCBDVP 530550 Watt Bifacial Module with Dual Glass 550W:

- Manufacturer (Brand): Jinko Solar
- Cell Type: Monocrystalline
- Cell Dimensions: 2278 x 1134 x 30mm
- Number of cells: 144.0
- Application: Commercial
- Minimum wattage: 550.0
- Weight: 14.51 kg

You can find more information about this product and make a purchase on the [Tiger Pro 72HC-BDVP. 530-550 Watt. Bifacial 
Module with Dual Glass (550W) product page](https://inxeptionmarketplace-crate.inxeption.io/purl/inxeptionenergymarketplace-tiger-pro-72hc-bdvp-530-550-watt-bifacial-module-with-dual-glass-550w-).
Do you want me to show alternative products based on weight or wattage?
                            \n\nContext: {context}\n\n"""
        #instructions = f"I want you to act as Information service provider, which is specifically design for Inxeption.Your task is to give answers and provide usefull in depth information.Answer the question based on the context below, try to understand the context,user can ask questions based on synonyms or by twiking sentences.Include all the URL avaialble in the context.if the question can't be answered based on the context, say \"I don't know\"If question is not based on the context then,You must ask questions before answering to understand better what I am seeking.\n\nContext: {context}\n\n"
        #prompt = f"I want you to act as Information provider, which is specifically design for Inxeption.Your task is to give answers and provide usefull in depth information.Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\" If answers are related to product then add Inx-prod: in the start of the response or else add inx_gen:\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
        #prompt = f"I want you to  act as information provider. I will provide you the context and your job is to answer the questions based on the provided context. Read context carefully and try to provide answers,I can tweak questions or I can ask contextual questions which is not exactly same to the context.If I am asking any question outside the scope of context just say I dont know.Keep in mind that Do not include anything related to provided context in the response. Act as if you dont know the context.\n\nContext: {context}\n\nQuestion: {question}"
        prompt = ''.join(prompt)
        #print(prompt)
        messages = [
        { "role": "system", "content": instructions },
        ]
        # add the previous questions and answers
        for question, answer in previous_questions_and_answers[-MAX_CONTEXT_QUESTIONS:]:
            messages.append({ "role": "user", "content": question })
            messages.append({ "role": "assistant", "content": answer })
        # add the new question
        messages.append({ "role": "user", "content": prompt })
        response = openai.ChatCompletion.create(
            
            model="gpt-3.5-turbo",
            messages = messages,
            max_tokens = 1024,
            temperature = 0)
                    
        message = response.choices[0].message.content
        #print(message)
            
        return message
        #return response
    except Exception as e:
        print(e)
        return ""

################################################################################
### Step 2
################################################################################
if __name__ == "__main__":
    previous_questions_and_answers = []
    #if len(sys.argv) < 2:
    #    print("Please provide an input argument.")
    #else:
        #try:
    
    #df=pd.read_csv('processed/embeddingv1_30.csv', index_col=0)
    #df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    df = pd.read_parquet('dataset_850prod_updated.parquet.gzip') ##working
    #question_test = sys.argv[1]
    #print(question_test)
    #start_time = time.time()
    while True:

        # ask the user for their question
        new_question = input(
            Fore.GREEN + Style.BRIGHT + "User: " + Style.RESET_ALL
        )
        replay = answer_question_2(df, question=new_question, debug=False)
        previous_questions_and_answers.append((new_question, replay))
    #print("##########################")
        #print(replay) 
        print(Fore.CYAN + Style.BRIGHT + "INX: " + Style.NORMAL +Fore.RED + Style.BRIGHT+ Style.NORMAL +replay)
    #end_time = time.time()
    #diff = end_time - start_time
    #print(diff)
        #except Exception as e:
            #print(e)

        




