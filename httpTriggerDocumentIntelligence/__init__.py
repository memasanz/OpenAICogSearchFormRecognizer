from azure.core.exceptions import ResourceNotFoundError
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient 
import azure.functions as func
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import json
import os
import openai
from OutputTables import OutputTables, OutputTable, TableCell
import pandas as pd
import re


#code 12/7

def push_to_vector_index(data, embeddings, source):
    logging.info('push_to_vector_index')
    search_keys = []
    service_endpoint = os.environ['COG_SEARCH_ENDPOINT']
    index_name = os.environ['COG_SEARCH_INDEX_NAME']
    key = os.environ['COG_SEARCH_KEY']
    credential = AzureKeyCredential(key)

    search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)
    title_embeddings = get_embedding(source)

    path = "https://" + os.environ['STORAGE_ACCOUNT'] + ".blob.core.windows.net/" + os.environ['STORAGE_ACCOUNT_CONTAINER'] + "/" + source
    path = path.replace(' ', '%20')
    
    try:
        docs = search_client.search(search_text=f"{source}", search_fields=["title"], include_total_count = True)
        count = docs.get_count()
        logging.info('total count retrieved from search = ' + str(count))
        delete_docs = []
        if count > 0:
            for x in docs:
                if x['path'] == path:
                    delete_docs.append({"key" : x['key']})


            if len(delete_docs) > 0:
                logging.info('about to delete documents')
                logging.info('delete_docs:' + str(len(delete_docs)))
                result = search_client.delete_documents(documents=delete_docs)
                for i in range (0, len(result)):
                    if result[0].succeeded  == False:
                        raise ValueError('A very specific bad thing happened.')
                logging.info('deletion occured:'  + str(len(result)))
        else:
            logging.info('no documents to delete')

        logging.info('about to upload documents')
        logging.info('about to upload documents:' + str(len(data)))
    except Exception as e:
        logging.info(e)
        logging.info('Error in search_client.search')
        pass


    for i in range(len(data)):
        text = data[i]
        title_embeddings = title_embeddings
        embedd = embeddings[i]
        random_str = source + "_" + str(i)
        random_str = re.sub(r'[\[\]\(\)\*\&\^\%\$\#\@\!\.]', '-', random_str)
        random_str = random_str.replace(" ", "-")

        logging.info(str(random_str))
        search_keys.append(str(random_str))
        #logging.info(len(title_embeddings))
        #logging.info(len(embedd))
        logging.info(random_str)
        
        document = {
            "key": f"{random_str}",
            "title": f"{source}",
            "content": f"{text}",
            "path": f"{path}",
            "contentVector": embedd,
            "titleVector": title_embeddings
        }
        logging.info("uploading document")
        logging.info(document)
        result = search_client.upload_documents(documents=document)
        logging.info("Upload of new document succeeded: {}".format(result[0].succeeded))
        logging.info('**************************************')
        json_string = json.dumps(document)
        #logging.info(json_string)
    return search_keys

def get_tables(result):
    myOutputTables = OutputTables()

    for i in range (0, len(result.tables)):

        table = result.tables[i]

        for j in range(0, len(table.bounding_regions)):
            region = table.bounding_regions[j]

            output_table = OutputTable(region.page_number, table.row_count, table.column_count)
            for c in range(0, len(table.cells)):
                cell = table.cells[c]

                output_cell = TableCell(cell.row_index, cell.column_index, cell.content, cell.row_span, cell.column_span)
                output_table.add_record(output_cell)
            myOutputTables.add_table(output_table)
        
    return myOutputTables

def get_text(page_number, result):
        #print('page number = ' + str(page_number))
        page = result.pages[page_number];
        #print(page)
        content = ''
        for line_idx, line in enumerate(page.lines):
            test = 5
            content = content + line.content + '/n'
        return content

def get_tables_by_page(Outputtables, page_number):  
    filtered_tables = []  
    for table in Outputtables.tables:  
        if table.page_number == page_number:  
            filtered_tables.append(table)  
    return filtered_tables

def get_client():
    endpoint = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")
    openai_type = os.getenv("OPENAI_API_TYPE", None)
    api_version = os.getenv("OPENAI_API_VERSION", None)
 
    if openai_type=='azure':
        client = openai.AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version
        )
        return client
    else:
        openai.api_key = api_key
        return None
    
def get_embedding(text):
    text = text.replace("\n", " ")
    model = os.getenv('TEXT_EMBEDDING_MODEL')
    client = get_client()
   
    embeddings = client.embeddings.create(input = [text], model=model).data[0].embedding
    return embeddings

def text_split_embedd(source):
    logging.info('text_split_embedd')

    endpoint = os.environ["FORMS_RECOGNIZER_ENDPOINT"]
    key = os.environ['FORMS_RECOGNIZER_KEY']
    formUrl = "https://" + os.environ['STORAGE_ACCOUNT'] + ".blob.core.windows.net/" + os.environ['STORAGE_ACCOUNT_CONTAINER'] + "/" + source
    formUrl = formUrl.replace(' ', '%20')
    logging.info(formUrl)

    logging.info('about to do document analysis')
    document_analysis_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    poller = document_analysis_client.begin_analyze_document_from_url("prebuilt-layout", formUrl)
    result = poller.result()
    logging.info('document analysis complete')
    #get all tables
    myOutputTables = get_tables(result)


    page_content = []
    for i in range(0, len(result.pages)):
        logging.info('***********************************')
        logging.info('in for loop, i = ' + str(i))
        content = get_text(i, result)
        logging.info(content)
        page_outputtables = get_tables_by_page(myOutputTables, i+1)
        for j in range(0, len(page_outputtables)):
            #content = '\n' + content + page_outputtables[j].to_json() 
            content = '\n' + '\n'  + content + page_outputtables[j].to_markdown()
        page_content.append(content)

    df = pd.DataFrame(page_content, columns =['text'])
    
    engine = os.environ['TEXT_EMBEDDING_MODEL']
    start = 0
    df_result = pd.DataFrame()
    for i, g in df.groupby(df.index // 1):
        try:
            g['curie_search'] = g["text"].apply(lambda x : get_embedding(x))

            df_result = pd.concat([df_result,g], axis=0)

        except Exception as e:
            logging.info(e)
            logging.info('Error in get_embedding')
            continue
        finally:
            continue

    df_result
    embeddings = []
    data = []

    for index, row in df_result.iterrows():
        embeddings.append(row['curie_search'])
        data.append(row['text'])

    logging.info('embeddings')
    for i in range(0, len(embeddings)):
        logging.info(embeddings[i])

    return data, embeddings


def push_to_vector_index(data, embeddings, source):
    logging.info('push_to_vector_index')
    try:
        search_keys = []
        service_endpoint = os.environ['COG_SEARCH_ENDPOINT']
        index_name = os.environ['COG_SEARCH_INDEX_NAME']
        key = os.environ['COG_SEARCH_KEY']
        credential = AzureKeyCredential(key)

        logging.info('got credentials')
        search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)
        title_embeddings = get_embedding(source)

        path = "https://" + os.environ['STORAGE_ACCOUNT'] + ".blob.core.windows.net/" + os.environ['STORAGE_ACCOUNT_CONTAINER'] + "/" + source
        path = path.replace(' ', '%20')
    
    
        docs = search_client.search(search_text=f"{source}", search_fields=["title"], include_total_count = True)
        count = docs.get_count()
        logging.info('total count retrieved from search = ' + str(count))
        delete_docs = []
        if count > 0:
            for x in docs:
                if x['path'] == path:
                    delete_docs.append({"key" : x['key']})


            if len(delete_docs) > 0:
                logging.info('about to delete documents')
                logging.info('delete_docs:' + str(len(delete_docs)))
                result = search_client.delete_documents(documents=delete_docs)
                for i in range (0, len(result)):
                    if result[0].succeeded  == False:
                        raise ValueError('A very specific bad thing happened.')
                logging.info('deletion occured:'  + str(len(result)))
        else:
            logging.info('no documents to delete')

        logging.info('about to upload documents')
        logging.info('about to upload documents:' + str(len(data)))
    except Exception as e:
        logging.info(e)
        logging.info('Error in search_client.search')
        pass


    for i in range(len(data)):
        text = data[i]
        title_embeddings = title_embeddings
        embedd = embeddings[i]
        random_str = source + "_" + str(i)
        random_str = re.sub(r'[\[\]\(\)\*\&\^\%\$\#\@\!\.]', '-', random_str)
        random_str = random_str.replace(" ", "-")

        logging.info(str(random_str))
        search_keys.append(str(random_str))
        logging.info(random_str)
        
        document = {
            "key": f"{random_str}",
            "index": f"{i}",
            "title": f"{source}",
            "content": f"{text}",
            "path": f"{path}",
            "contentVector": embedd,
            "titleVector": title_embeddings
        }
        logging.info("uploading document")
        logging.info(document)
        result = search_client.upload_documents(documents=document)
        logging.info("Upload of new document succeeded: {}".format(result[0].succeeded))
        logging.info('**************************************')
        json_string = json.dumps(document)
    return search_keys

def compose_response(json_data):
    logging.info('in compose response')
    body  = json.loads(json_data)
    assert ('values' in body), "request does not implement the custom skill interface"
    values = body['values']
    # Prepare the Output before the loop
    results = {}
    results["values"] = []

    endpoint = os.environ["FORMS_RECOGNIZER_ENDPOINT"]
    logging.info(endpoint)
    key = os.environ["FORMS_RECOGNIZER_KEY"]
    logging.info(key)
    for value in values:
        output_record = transform_value(value)
        if output_record != None:
            results["values"].append(output_record)
            break
    return json.dumps(results, ensure_ascii=False)

def transform_value(value):
    logging.info('in transform_value')
    try:
        recordId = value['recordId']
    except AssertionError  as error:
        return None

    # Validate the inputs
    try:  
        logging.info(value)       
        assert ('data' in value), "'data' field is required."
        data = value['data']   
    except AssertionError  as error:
        return (
            {
            "recordId": recordId,
            "data":{},
            "errors": [ { "message": "Error:" + error.args[0] }   ]
            })
    try:   
        logging.info('about to get source')             
        source = value['data']['source']

        logging.info('source')
        logging.info(source)

        API_BASE = os.environ["OPENAI_API_BASE"]
        API_KEY = os.environ["OPENAI_API_KEY"]
        API_VERSION = os.environ["OPENAI_API_VERSION"]
        API_TYPE = os.environ["OPENAI_API_TYPE"]
        

        openai.api_type = API_TYPE
        openai.api_base = API_BASE
        openai.api_version = API_VERSION
        openai.api_key  = API_KEY

        data, embeddings = text_split_embedd(source)

        logging.info('push to vector index')
        vector_search_keys = push_to_vector_index(data, embeddings, source)


    except Exception as e:
        logging.info(e)
        return (
            {
            "recordId": recordId,
            "errors": [ { "message": "Could not complete operation for record."  } , {e}  ]
            })

    return ({
            "recordId": recordId,
            "data": {
                "embeddings_text": data,
                "embeddings": embeddings,
                "vector_search_keys": vector_search_keys
                    }
            })

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Invoked AnalyzeForm Skill.')
    try:
        body = json.dumps(req.get_json())
        logging.info('body')
        logging.info(body)
        if body:
            result = compose_response(body)
            return func.HttpResponse(result, mimetype="application/json")
        else:
            return func.HttpResponse(
                "The body of the request could not be parsed",
                status_code=400
            )
    except ValueError:
        return func.HttpResponse(
             "The body of the request could not be parsed",
             status_code=400
        )
    except KeyError:
        return func.HttpResponse(   
             "Skill configuration error. Endpoint, key and model_id required.",
             status_code=400
        )
    except AssertionError  as error:
        return func.HttpResponse(   
             "Request format is not a valid custom skill input",
             status_code=400
        )