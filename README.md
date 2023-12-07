## Azure Cognitive Seach with a Custom Skill and Azure Open AI

### Note this repo only works with supported data types for the layout api for form recognizer (pdfs, etc)
### Required components:
- Blob Storage (with a container holding your docs)
- Function App (be sure to set your application settings)
- Cognitive Search (be sure to enable semantic search on the resource)
- Document Intelligence or Azure AI Multi-Service Account (turn on System Assigned Managed Identity - and provide access to storage account at the container level "Storage Blob Data Reader". )

### .env file for running the notebook
In your .env file include the following parameters:

```
FORMS_RECOGNIZER_ENDPOINT = 'https://<xxxx>.cognitiveservices.azure.com/'
FORMS_RECOGNIZER_KEY = '<yourkeygoeshere>'

COG_SEARCH_ENDPOINT = 'https://<search-resource>.search.windows.net'
COG_SEARCH_INDEX_NAME = 'index-name' 
COG_SEARCH_KEY = 'ODdaflxDWvvXXXXXqFqztzWxGyF73'

TEXT_EMBEDDING_MODEL = 'text-embedding-ada-002'
OPENAI_API_BASE = 'https://xxxxxx.openai.azure.com'
OPENAI_API_KEY = '<yourkeygoeshere>'
OPENAI_API_VERSION = '2023-07-01-preview'
OPENAI_API_TYPE = 'azure'

STORAGE_ACCOUNT = '<storage-account here>'
STORAGE_ACCOUNT_CONTAINER = '<container holding docs>'
STORAGE_CONNECTION_STRING = '<ConnectionString>'

functionAppUrlAndKey = 'https://<azurefunctionapp>.azurewebsites.net/api/httpTriggerDocumentIntelligence?code=<keygoeshere>'
```

### Application Settings 
#### for Your Deployed function app (the stuff getting info from os.getenv)

```
FORMS_RECOGNIZER_ENDPOINT
FORMS_RECOGNIZER_KEY

COG_SEARCH_ENDPOINT
COG_SEARCH_INDEX_NAME
COG_SEARCH_KEY

TEXT_EMBEDDING_MODEL
OPENAI_API_BASE
OPENAI_API_KEY
OPENAI_API_VERSION
OPENAI_API_TYPE

STORAGE_ACCOUNT
STORAGE_ACCOUNT_CONTAINER
```
