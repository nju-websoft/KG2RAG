from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from tqdm import tqdm
import argparse

from FlagEmbedding import FlagReranker
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings,VectorStoreIndex,PromptTemplate,StorageContext,load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.embeddings.ollama import OllamaEmbedding
from util.kg_post_processor import NaivePostprocessor,KGRetrievePostProcessor,ngram_overlap,GraphFilterPostProcessor,KGIntraInterPostProcessor
from util.kg_response_synthesizer import get_response_synthesizer

def kg_rag_parallel(data, doc2kg, top_k=5, workers=4,persist_dir=None,reranker='../model/bge-reranker-large'):
    prediction = {'answer': {}, 'sp': {}}

    doc_chunks = []
    chunks_index = dict()
    ents = set()
    for sample in data:
        for ctx in sample['context']:
            ent = ctx[0]
            ents.add(ent)
            if ent not in chunks_index:
                chunks_index[ent] = dict()
            for i in range(len(ctx[1])):
                doc_chunk = TextNode(text=f'{ent}: {ctx[1][i]}',id_=f'{ent}##{str(i)}')
                doc_chunks.append(doc_chunk)
                if str(i) not in chunks_index[ent]:
                    chunks_index[ent][str(i)] = doc_chunk.text
    if (persist_dir is not None) and os.path.exists(persist_dir):
        print('Load index from persist dir')
        sc = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(sc)
    else:
        print('Create and save index to persist dir')
        index = VectorStoreIndex(doc_chunks,show_progress=True)
        if persist_dir is not None:
            os.makedirs(persist_dir,exist_ok=True)
            index.storage_context.persist(persist_dir=persist_dir)
    print(f'Index ready in persist dir {persist_dir}')
    retriever = VectorIndexRetriever(index=index,similarity_top_k=top_k)
    qa_rag_template_str = 'Context information is below.\n{context_str}\nGive a short factoid answer (as few words as possible).\nQ: Were Scott Derrickson and Ed Wood of the same nationality?\nA: Yes.\nQ: Who was born earlier, Emma Bull or Virginia Woolf?\nA: Adeline Virginia Woolf.\nQ: The arena where the Lewiston Maineiacs played their home games can seat how many people?\nA: 3,677 seated.\nQ: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?\nA: Chief of Protocol.\n---------------------\nQ: {query_str}\nA: '
    qa_rag_prompt_template = PromptTemplate(qa_rag_template_str)
    response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT,text_qa_template=qa_rag_prompt_template)
    
    kg_post_processor1 = KGRetrievePostProcessor(ents=ents,doc2kg=doc2kg,chunks_index=chunks_index)
    bge_reranker = FlagReranker(model_name_or_path=reranker)
    kg_post_processor2 = GraphFilterPostProcessor(topk=top_k,ents=ents,doc2kg=doc2kg,chunks_index=chunks_index,reranker=bge_reranker)

    engine = RetrieverQueryEngine(retriever=retriever,response_synthesizer=response_synthesizer,node_postprocessors=[kg_post_processor1,kg_post_processor2,NaivePostprocessor()])

    test_size = len(data)

    sps_count = []
    for sample in tqdm(data[:min(len(data),test_size)]):
        sample_id = sample['_id']
        sample_question = sample['question']
        sample_answer = sample['answer']

        response = engine.query(sample_question)
        answer = response.response
        sps = [[source_node.node.id_.split('##')[0], int(source_node.node.id_.split('##')[1])] for source_node in response.source_nodes]
        prediction['answer'][sample_id] = answer
        prediction['sp'][sample_id] = sps
        sps_count.append(len(sps))

    print(f'Avg #sps: {sum(sps_count)/len(sps_count)}')

    return prediction

def main(args):
    data_path = args.data_path
    with open(data_path,'r',encoding='utf-8') as f:
        data = json.load(f)

    ents = set()
    for sample in data:
        for ctx in sample['context']:
            ents.add(ctx[0])

    kg_dir = args.kg_dir
    doc2kg = dict()
    print(f'\n{"-"*20}\nLoading KGs')
    for ent in tqdm(ents):
        subkg_path = os.path.join(kg_dir,f'{ent.replace("/","_")}.json')
        if not os.path.exists(subkg_path):
            continue
        with open(subkg_path,'r',encoding='utf=8')as fin:
            subkg = json.load(fin)
            if subkg and len(subkg.keys())>0:
                for seq in subkg.keys():
                    for triplet in subkg[seq]:
                        h,r,t = triplet
                        if (ngram_overlap(h,ent)>=0.90) or (ngram_overlap(ent,h)>=0.90):
                            h = ent
                        if (ngram_overlap(t,ent)>=0.90) or (ngram_overlap(ent,t)>=0.90):
                            t = ent
                        triplet = h,r,t
                    if len(subkg[seq])==0:
                        del subkg[seq]
                if len(subkg.keys())>0:
                    doc2kg[ent] = subkg

    model_name = args.model_name
    print('Init Ollama model')
    Settings.llm = Ollama(model=model_name,request_timeout=200)
    embed_model_name = args.embed_model_name
    print('Init Ollama embedding')
    Settings.embed_model = OllamaEmbedding(model_name=embed_model_name)
    top_k = args.top_k
    workers = args.num_workers
    persist_dir = args.persist_dir
    reranker = args.reranker
    prediction = kg_rag_parallel(data,doc2kg,top_k=top_k,workers=workers,persist_dir=persist_dir,reranker=reranker)

    result_path = args.result_path
    with open(result_path,'w',encoding='utf-8') as f:
        json.dump(prediction,f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hotpot full
    parser.add_argument('--data_path',type=str,default='../data/hotpotqa/hotpot_dev_distractor_v1.json',help='Path to the data file')
    parser.add_argument('--result_path',type=str,default='../output/hotpot/hotpot_dev_distractor_v1_full.json',help='Path to the result file')
    parser.add_argument('--kg_dir',type=str,default='../data/hotpotqa/kgs/extract_subkgs')
    parser.add_argument('--persist_dir',type=str,default='../data/ollama_index/hotpotqa',help='Directory to store the index')

    # # pu-hotpot full
    # parser.add_argument('--data_path',type=str,default='../data/pu-hotpotqa/hotpot_dev_distractor_v1.json',help='Path to the data file')
    # parser.add_argument('--result_path',type=str,default='../output/pu-hotpot/pu-hotpot_dev_distractor_v1_full.json',help='Path to the result file')
    # parser.add_argument('--kg_dir',type=str,default='../data/pu-hotpotqa/kgs/extract_subkgs')
    # parser.add_argument('--persist_dir',type=str,default='../data/ollama_index/vector_pu_hotpotqa',help='Directory to store the index')

    parser.add_argument('--model_name',type=str,default='llama3:8b',help='Ollama model name')
    parser.add_argument('--embed_model_name',type=str,default='mxbai-embed-large',help='Ollama embedding model name for indexing')
    parser.add_argument('--top_k',type=int,default=10,help='Top k similar documents')
    parser.add_argument('--num_workers',type=int,default=4,help='Number of workers for parallel processing')
    parser.add_argument('--reranker',type=str,default='../model/bge-reranker-large',help='Path of the reranker model')
    args = parser.parse_args()
    main(args)