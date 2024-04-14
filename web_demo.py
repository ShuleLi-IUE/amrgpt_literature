#!/usr/bin/env python
# coding=utf-8
import gradio as gr
from openai_utils import init_openai, get_completion_openai_stream
from prompt_utils import build_prompt
from vectordb_utils_paper import PaperVectorDB
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 
import os
import pandas as pd
import time
import sys
import errno
import pickle
from logger import log_info, log_debug, log_warning

top_n = 8
recall_n = 80
distance = "l2"
batch_size = 12
num_workers = None
vec_db_paper = PaperVectorDB(space=distance,
                             batch_size=batch_size)
rerank_model = CrossEncoder(os.getenv('RERANK_MODEL_PATH'))
index_path = os.getenv('PAPER_INDEX_PATH')

def init_db_load_index():
    global vec_db_paper
    init_openai()
    log_info("---init database by index file begin---")
    log_info(f"index path: {index_path}")
    with open(index_path, 'rb') as file:
        vec_db_paper = pickle.load(file)
    
    log_info("---init database by index file end---")
    
def search_db(user_input, chatbot, context, search_field, source_type, mode_type):
    log_info("---chat button---")
    
    if mode_type == "Efficiency":
        log_info("===hnsw===")
        search_labels = vec_db_paper.search_bge(user_input, top_n)

        titles, abstracts, journals, authors, citations, years, links, DOIs = vec_db_paper.get_context_by_labels(search_labels)
        res = [titles[i] + '. ' + abstracts[i].strip("'") for i in range(top_n)]
        
        search_field = "\n\n".join([f"{i+1}. [Reference: {authors[i]}. ({years[i]}). {titles[i]}. {journals[i]}. {DOIs[i]}.]\n{abstracts[i]}" for i in range(top_n)])
        prompt = build_prompt(source_type=source_type, info=[f"{res[i]} [Reference: {titles[i]}, {years[i]}, {journals[i]}]" for i in range(top_n)], query=user_input)
        
    elif mode_type == "Accuracy":
        log_info("===rerank===")
        scores, titles, abstracts, journals, authors, citations, years, links, DOIs = rerank(user_input, top_n, recall_n, source_type)
        res = [titles[i] + '. ' + abstracts[i].strip("'") for i in range(top_n)]

        search_field = "\n\n".join([f"{i+1}. [Reference: {authors[i]}. ({years[i]}). {titles[i]}. {journals[i]}. {DOIs[i]}.]\n{abstracts[i]}" for i in range(top_n)])
        prompt = build_prompt(source_type=source_type, info=[f"{res[i]} [Reference: {titles[i]}, {years[i]}, {journals[i]}]" for i in range(top_n)], query=user_input)
    
 
    log_info(f"source_type: {source_type}, mode_type: {mode_type}\nprompt content built:\n{prompt}")
    return prompt, search_field


def rerank(user_input, top_n, recall_n, source_type):
    search_labels = vec_db_paper.search_bge(user_input, recall_n)
    t0 = time.time()
    titles, abstracts, journals, authors, citations, years, links, DOIs = vec_db_paper.get_context_by_labels(search_labels)
    t1 = time.time()
    log_info(f"vec_db_shule.get_context_by_labels costs: {t1 - t0}")

    documents = [titles[i] + '. ' + abstracts[i] for i in range(top_n)]
    res = rerank_model.rank(documents = documents,
                            query=user_input,
                            batch_size = 1,
                            return_documents = False,
                            show_progress_bar = False)
    t2 = time.time()
    log_info(f"rerank_model.predict costs: {t2 - t1}")

    ids = [i['corpus_id'] for i in res][:top_n]
    scores = [i['score'] for i in res][:top_n]
    log_info(f"finish rerank {recall_n} texts, return highest {top_n} texts")
    
    titles, abstracts, journals, authors, citations, years, links, DOIs
    return scores, [titles[i] for i in ids],  [abstracts[i] for i in ids], [journals[i] for i in ids], [authors[i] for i in ids], [citations[i] for i in ids], [years[i] for i in ids], [links[i] for i in ids], [DOIs[i] for i in ids]
    # search_field = "\n\n".join([f"{i+1}. [Reference: {authors[i]}. ({years[i]}). {titles[i]}. {journals[i]}. {DOIs[i]}. {links[i]}. Citations: {citations[i]}]\n{abstracts[i]}" for i in range(top_n)])
    # prompt = build_prompt(source_type=source_type, info=[f"{res[i]} [Reference: {titles[i]}, {years[i]}, {journals[i]}]" for i in range(top_n)], query=user_input)
    # return search_field, prompt
    

def reset_state():
    log_info("---reset state---")
    return [], [], "", ""


def main():
    log_info("===begin gradio===")
    with gr.Blocks(css="web_css.css") as demo:
        with gr.Row() as output_field:
            with gr.Column() as chat_col:
                chatbot = gr.Chatbot(height=450, show_label=True, label="Chatbot")
            with gr.Column() as ref_col:
                search_field = gr.TextArea(show_label=True, label="Reference", placeholder="Reference...", elem_classes="box_height", container=False, lines=50)

        with gr.Column(elem_classes=".input_field") as input_field:
            with gr.Row(elem_classes=".dropdown_group"):
                model = gr.Dropdown(label="Model", choices=["GPT-3.5", "GPT-4"], value="GPT-4", filterable=False, min_width=50)
                source = gr.Dropdown(label="Source", choices=["Hybrid", "Standalone"], value="Hybrid", filterable=False)
                mode = gr.Dropdown(label="Mode", choices=["Accuracy", "Efficiency"], value="Accuracy", filterable=False)
                history = gr.Dropdown(label="History", choices=["Yes", "No"], value="Yes", filterable=False)
            with gr.Row():
                user_input = gr.Textbox(show_label=False, placeholder="Enter your questions about AMR...", lines=3)
            with gr.Row():
                submitBtn = gr.Button("Submit", variant="primary")
                emptyBtn = gr.Button("Clear")

        context = gr.State([])

        def user(user_message, history):
            return user_message, history + [[user_message, None]]
        
        def bot2(user_input, chatbot, context, search_field, model, source, mode, history):
            print(model, source, mode, history, user_input)
            log_info(f"model: {model}, source: {source}, mode: {mode}, history: {history}\nuser_input:{user_input}")
            prompt, search_field = search_db(user_input, chatbot, context, search_field, source, mode)
            
            # clear user input
            user_input = ""

            # print("prompt and search_field:", prompt, search_field)
            log_info("===get completion===")
            response_stream = get_completion_openai_stream(prompt, context, model, history)
            response = ""
            chatbot[-1][1] = ""
            for word in response_stream:
                chatbot[-1][1] += word
                response += word
                yield user_input, chatbot, context, ""
            
            context.append({'role': 'user', 'content': user_input})
            context.append({'role': 'assistant', 'content': response})
            log_info(f"completion content:\n{response}")

            # response is empty, lead to a error in gradio
            # put a netword error to client
            if chatbot[-1][1] == "":
                chatbot[-1][1] += "Network Error. Wait seconds and try again."
                search_field = ""
            
            yield user_input, chatbot, context, search_field

        submitBtn.click(user, [user_input, chatbot],
                        [user_input, chatbot], queue=False
                        ).then(
                            bot2, 
                            [user_input, chatbot, context, search_field, model, source, mode, history], 
                            [user_input, chatbot, context, search_field]
                        )
        emptyBtn.click(reset_state, outputs=[chatbot, context, user_input, search_field])

    demo.queue().launch(share=False, server_name='0.0.0.0', server_port=8887, inbrowser=False, show_api=False)
    
    
if __name__ == "__main__":
    init_db_load_index()
    main()