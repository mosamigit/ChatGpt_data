#!/usr/bin/env python
# coding: utf-8
#spectrum.app@inxeption.com / Spectrum4ChatGPT!
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
import json
from colorama import Fore, Back, Style
import ast 
################################################################################
### Step 1
################################################################################                                                                                                                                                                                                   


#load_dotenv(Path(r"C:\Users\norin.saiyed\web-crawl-q-and-a\.env"))
load_dotenv()
#api_key = os.environ["API_KEY"]
openai.api_key = 'sk-kZAcnb4xH3LbPHI4n2sWT3BlbkFJvuC8r'
#openai.api_key = api_key
def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """
    #f'text-search-{size}-doc-001'
    #text-embedding-ada-002

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
        prompt= f" \n\nContext: {context}\n\n Question: {question}\nAnswer:"
        #print(prompt)
        # instructions = f"""I want you to act as customer support chatbot, which is specifically design for Inxeption.Your task is to provide information and suggestions related to solar panels.
        #                     If user is asking for solar panel which is not in the context then do not provide any false information.
        #                     Answer the question based on the context below, try to understand the context,user can ask questions based on synonyms or by twiking sentences.
        #                     If you are unable to provide an answer based on the context, please respond with the phrase \" I can't help with that.\".
        #                     If question is not based on the context then,You must ask questions before answering to understand better what I am seeking.
							
		# 					Answer should be authenticated and should provide true information related to products and specifically URL. Do not include any URL which is not there in the context.
        #                     user can ask questions based on specification such as cell dimension,weight,number of cells.
        #                     please double check the wattage which user is asking in the question. If you do not have solar panel specific to that wattage then simply say "we dont have".
        #                     If user is asking for specific wattage based solar panel then search for the context and thenprovide the product URL.Do not provide false URL.
                            
        #                     If answer contains a sequence of instructions ,rewrite those instruction in the following format: step1-...
        #                     If user is asking anything from the context related to number of cells,cell dimensions,wattage then provide exact product that matches that information.do not provide any wrong product name. for eg,Questions:The panel has 132 cells and measures 1855 x 1029 x 30mm in cell dimensions
        #                     Answer:The Tiger N Type 66TR 390-410 Watt MonoFacial Module 400W has 132 cells and measures 1855 x 1029 x 30mm in cell dimensions.
        #                     If you are interested in purchasing this panel or have any further questions, you can visit the product page on the Inxeption Energy Marketplace: [Tiger N Type 66TR. 390-410 Watt. Mono-Facial Module. (400W)]("https://inxeptionmarketplace-crate.inxeption.io/purl/inxeptionenergymarketplace-tiger-n-type-66tr-390-410-watt-mono-facial-module-400w-").

        #                     \n\n
        #                     Ideal complete response related to specific  product should be:
        #                     \n\n
        #                     Certainly! The Tiger Pro 72HCBDVP 530-550 Watt Bifacial Module with Dual Glass (550W) is a high-efficiency monocrystalline solar panel manufactured by Jinko Solar.
        #                     The panel has 144 cells and measures 2278 x 1134 x 30mm in cell dimensions.
        #                     It has a nominal wattage power of 550 watts.
        #                     The panel weighs 14.51kg and suitable for Commercial applications.
        #                     If you are interested in purchasing this panel or have any further questions, you can visit the product page on the Inxeption Energy Marketplace:
        #                     [Tiger Pro 72HCBDVP 530-550 Watt Bifacial Module with Dual Glass (550W)](https://inxeptionmarketplace-crate.inxeption.io/purl/inxeptionenergymarketplace-tiger-pro-72hc-bdvp-530-550-watt-bifacial-module-with-dual-glass-550w-)
                            
        #                     Need assistance with a custom order? We can help you find the right solutions for your business. Request a quote or call us at 888.852.4783 and select option 4.
        #                     Please let me know if you have any other questions or if there is anything else I can help you with.
        #                     \n\n
        #                     ideal respose related to list of product should be
        #                     Questions:list all Tiger products
        #                     Answer:Here are some of the Tiger products available on the Inxeption Energy Marketplace:
        #                     - [Tiger Pro 72HC-BDVP. 530-550 Watt. Bifacial Module with Dual Glass (550W)](https://inxeptionmarketplace-crate.inxeption.io/purl/inxeptionenergymarketplace-tiger-pro-72hc-bdvp-530-550-watt-bifacial-module-with-dual-glass-550w-)
        #                     - [Tiger N Type 66TR. 390-410 Watt. Mono-Facial Module.  (400W)](https://inxeptionmarketplace-crate.inxeption.io/purl/inxeptionenergymarketplace-tiger-n-type-66tr-390-410-watt-mono-facial-module-400w-)
        #                     - [Tiger 78TR. 470-490 Watt. Mono-Facial Module.  (490W)](https://inxeptionmarketplace-crate.inxeption.io/purl/inxeptionenergymarketplace-tiger-78tr-470-490-watt-mono-facial-module-490w-)
        #                     \n\n
                            
        #                     If user is asking anything related to shipping cost of products other than solar panel and refrigerated products say please contact inxeption.
        #                     Add extra information related to best manufacturer brands which inxeption has along with the answers realted to brands.
                           
        #                    """

        instructions = f"""I want you to act as customer support chatbot, which is specifically design for Inxeption.Your task is to provide information and suggestions related to solar panels.
                            If user is asking for solar panel which is not in the context then do not provide any false information.
                            Answer the question based on the context below, try to understand the context,user can ask questions based on synonyms or by twiking sentences.
                            If you are unable to provide an answer based on the context, please respond with the phrase \" I can't help with that.\".
                            If question is not based on the context then,You must ask questions before answering to understand better what I am seeking.
							
			                Answer should be authenticated and should provide true information related to products and specifically URL. Do not include any URL which is not there in the
                            context.
                            user can ask questions based on specification such as No. of Shipments,No. of Consignees,Weight (Kg),Quantity,country,Top Products,HS code,Loading 
                            Port,Unloading Port.
                            please double check the features which user is asking in the question. If you do not have specific information to that feature then simply say "we dont have".
                            If user is asking for specific Month and year based import then search for the context and then provide the URL.Do not provide false URL.
                            
                            If answer contains a sequence of instructions ,rewrite those instruction in the following format: step1-...
                            If user is asking anything from the context related to No. of Shipments,No. of Consignees,Weight (Kg),Quantity,country,Top Products,HS code,Loading 
                            Port,Unloading Port then provide exact product that matches that information.do not provide any wrong product name. for eg,Questions:list of  consignee's in 
                            march 2023
                            Answer:Here are some of the consignees available on the Inxeption Energy Marketplace for solar panels:

			                - BYD AMERICA LLC
			                - FORT BEND SOLAR LLC
			                - VIETNAM SUNERGY JOINT STOCK COMPANY
			                - ASTRONERGY SOLAR
			                - NUSA SOLAR LLC
			                - VSUN SOLAR USA INC
			                You can find a complete list of consignees by clicking on this link: [Consignee](https://inxeption.com/search-results/trade-data?s=solar+panel)

                           
                            \n\n
                            Ideal complete response related to specific shipments or consignments in March 2023 should be:
                            \n\n
                            Certainly! The details description is as follows 
                            name:DELTA S EDGE SOLAR LLC;     
                            Address: 685 CR 219 GREENWOOD MISSISSIPPI; 
                            Country:USA;   
                            No.of Shipments:109;    
                            Top Products:SOLAR PANEL;    
                            Top Shippers:MUNDRA SOLAR ENERGY LIMITED,MUNDRA SOLAR PV LIMITED;  
                            Arrival Date(latest):01/25/23;     
                            Shipper:MUNDRA SOLAR ENERGY LIMITED;  
                            Shipper Address:SURVEY NO 180/P & OTHERS VILLAGE TUNDA;  
                            Loading Port:53306, MUNDRA;   
                            Unloading Port:1601, CHARLESTON, SC;    
                            Product Description:SOLAR PANEL;   
                            HS Code:8419.19;    
                            Quantity:1,240 PKG;   
                            Weight (Kg):36,960;      
                            For more details please check url:[DELTA S EDGE SOLAR LLC](https://inxeption.com/search-results/trade-data?s=solar+panel)
                         
                           
                           
                            # Need assistance with a custom order? We can help you find the right solutions for your business. Request a quote or call us at 888.852.4783 and select option 4.
                            # Please let me know if you have any other questions or if there is anything else I can help you with.
                            # \n\n
                            # ideal respose related to list of product should be
                            # Questions:list all Tiger products
                            # Answer:Here are some of the Tiger products available on the Inxeption Energy Marketplace:
                            # - [Tiger Pro 72HC-BDVP. 530-550 Watt. Bifacial Module with Dual Glass (550W)](https://inxeptionmarketplace-crate.inxeption.io/purl/inxeptionenergymarketplace-
                            #   tiger-pro-72hc-bdvp-530-550-watt-bifacial-module-with-dual-glass-550w-)
                            # - [Tiger N Type 66TR. 390-410 Watt. Mono-Facial Module.  (400W)](https://inxeptionmarketplace-crate.inxeption.io/purl/inxeptionenergymarketplace-tiger-n-
                            #   type-66tr-390-410-watt-mono-facial-module-400w-)
                            # - [Tiger 78TR. 470-490 Watt. Mono-Facial Module.  (490W)](https://inxeptionmarketplace-crate.inxeption.io/purl/inxeptionenergymarketplace-tiger-78tr-470-490-
                            #   watt-mono-facial-module-490w-)
                            # \n\n
                            
                            If user is asking anything related to shipping cost of products other than solar panel and refrigerated products say please contact inxeption.
                            Add extra information related to best manufacturer brands which inxeption has along with the answers realted to brands.
                           
                           """
        response = openai.ChatCompletion.create(
            
            model="gpt-3.5-turbo",
            messages = [{ "role": "system", "content": instructions},{"role": "user", "content": prompt}],
            max_tokens = 1024,
            temperature = 0,
            stream = True)
        #message = response.choices[0].message.content
        #print(message)
        collected_chunks = []
        collected_messages = []
        # iterate through the stream of events
        for chunk in response:
            #chunk_time = time.time() - start_time  # calculate the time delay of the chunk
            collected_chunks.append(chunk)  # save the event response
            chunk_message = chunk['choices'][0]['delta']  # extract the message
            #collected_messages.append(chunk_message)  # save the message
            #print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message}")  # print the delay and text
            #print(chunk_message)
            if "content" in chunk_message:
                message_text = chunk_message['content']
                #yield message_text
                print(Fore.RED + Style.BRIGHT +message_text, end='',flush=True)
                #print(Fore.CYAN + Style.BRIGHT + "INX: " + Style.NORMAL +Fore.RED + Style.BRIGHT+ Style.NORMAL +message_text)
                #print(f"{message_text}", end="")
                #collected_messages += message_text
            #time.sleep(1)
                
        # print the time delay and text received
        #print(f"Full response received {chunk_time:.2f} seconds after request")
        full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
        #print(f"Full conversation received: {full_reply_content}")
    except Exception as e:
        print(e)
        return ""

################################################################################
### Step 2
################################################################################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide an input argument.")
    else:
        try:
            
        
            #df = pd.read_parquet('processed/dataset_850prod.parquet.gzip') ##working
            df = pd.read_parquet('dataset_850prod_updated.parquet.gzip') ##working
            question_test = sys.argv[1]
            start_time = time.time()
            replay = answer_question_2(df, question=question_test, debug=False)
            #print("##########################")
            #print(replay)
            #end_time = time.time()
            #diff = end_time - start_time
            #print(diff)
        except Exception as e:
            print(e)

        




