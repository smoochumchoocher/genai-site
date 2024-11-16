{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "48ade4da-fafe-4bd7-b008-9eb1e26a2e82",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: streamlit in /opt/anaconda3/lib/python3.12/site-packages (1.32.0)\n",
      "Requirement already satisfied: langchain in /opt/anaconda3/lib/python3.12/site-packages (0.3.7)\n",
      "Collecting langchain_community\n",
      "  Downloading langchain_community-0.3.7-py3-none-any.whl.metadata (2.9 kB)\n",
      "Requirement already satisfied: openai in /opt/anaconda3/lib/python3.12/site-packages (1.54.4)\n",
      "Requirement already satisfied: tiktoken in /opt/anaconda3/lib/python3.12/site-packages (0.8.0)\n",
      "Requirement already satisfied: altair<6,>=4.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (5.0.1)\n",
      "Requirement already satisfied: blinker<2,>=1.0.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (1.6.2)\n",
      "Requirement already satisfied: cachetools<6,>=4.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (5.3.3)\n",
      "Requirement already satisfied: click<9,>=7.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (8.1.7)\n",
      "Requirement already satisfied: numpy<2,>=1.19.3 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (1.26.4)\n",
      "Requirement already satisfied: packaging<24,>=16.8 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (23.2)\n",
      "Requirement already satisfied: pandas<3,>=1.3.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (2.2.2)\n",
      "Requirement already satisfied: pillow<11,>=7.1.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (10.3.0)\n",
      "Requirement already satisfied: protobuf<5,>=3.20 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (3.20.3)\n",
      "Requirement already satisfied: pyarrow>=7.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (14.0.2)\n",
      "Requirement already satisfied: requests<3,>=2.27 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (2.32.2)\n",
      "Requirement already satisfied: rich<14,>=10.14.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (13.3.5)\n",
      "Requirement already satisfied: tenacity<9,>=8.1.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (8.2.2)\n",
      "Requirement already satisfied: toml<2,>=0.10.1 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (0.10.2)\n",
      "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (4.11.0)\n",
      "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (3.1.37)\n",
      "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (0.8.0)\n",
      "Requirement already satisfied: tornado<7,>=6.0.3 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (6.4.1)\n",
      "Requirement already satisfied: PyYAML>=5.3 in /opt/anaconda3/lib/python3.12/site-packages (from langchain) (6.0.1)\n",
      "Requirement already satisfied: SQLAlchemy<3,>=1.4 in /opt/anaconda3/lib/python3.12/site-packages (from langchain) (2.0.30)\n",
      "Requirement already satisfied: aiohttp<4.0.0,>=3.8.3 in /opt/anaconda3/lib/python3.12/site-packages (from langchain) (3.9.5)\n",
      "Requirement already satisfied: langchain-core<0.4.0,>=0.3.15 in /opt/anaconda3/lib/python3.12/site-packages (from langchain) (0.3.19)\n",
      "Requirement already satisfied: langchain-text-splitters<0.4.0,>=0.3.0 in /opt/anaconda3/lib/python3.12/site-packages (from langchain) (0.3.2)\n",
      "Requirement already satisfied: langsmith<0.2.0,>=0.1.17 in /opt/anaconda3/lib/python3.12/site-packages (from langchain) (0.1.143)\n",
      "Requirement already satisfied: pydantic<3.0.0,>=2.7.4 in /opt/anaconda3/lib/python3.12/site-packages (from langchain) (2.9.2)\n",
      "Collecting dataclasses-json<0.7,>=0.5.7 (from langchain_community)\n",
      "  Downloading dataclasses_json-0.6.7-py3-none-any.whl.metadata (25 kB)\n",
      "Collecting httpx-sse<0.5.0,>=0.4.0 (from langchain_community)\n",
      "  Downloading httpx_sse-0.4.0-py3-none-any.whl.metadata (9.0 kB)\n",
      "Collecting pydantic-settings<3.0.0,>=2.4.0 (from langchain_community)\n",
      "  Downloading pydantic_settings-2.6.1-py3-none-any.whl.metadata (3.5 kB)\n",
      "Requirement already satisfied: anyio<5,>=3.5.0 in /opt/anaconda3/lib/python3.12/site-packages (from openai) (4.2.0)\n",
      "Requirement already satisfied: distro<2,>=1.7.0 in /opt/anaconda3/lib/python3.12/site-packages (from openai) (1.9.0)\n",
      "Requirement already satisfied: httpx<1,>=0.23.0 in /opt/anaconda3/lib/python3.12/site-packages (from openai) (0.27.0)\n",
      "Requirement already satisfied: jiter<1,>=0.4.0 in /opt/anaconda3/lib/python3.12/site-packages (from openai) (0.7.1)\n",
      "Requirement already satisfied: sniffio in /opt/anaconda3/lib/python3.12/site-packages (from openai) (1.3.0)\n",
      "Requirement already satisfied: tqdm>4 in /opt/anaconda3/lib/python3.12/site-packages (from openai) (4.66.4)\n",
      "Requirement already satisfied: regex>=2022.1.18 in /opt/anaconda3/lib/python3.12/site-packages (from tiktoken) (2023.10.3)\n",
      "Requirement already satisfied: aiosignal>=1.1.2 in /opt/anaconda3/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.2.0)\n",
      "Requirement already satisfied: attrs>=17.3.0 in /opt/anaconda3/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (23.1.0)\n",
      "Requirement already satisfied: frozenlist>=1.1.1 in /opt/anaconda3/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.4.0)\n",
      "Requirement already satisfied: multidict<7.0,>=4.5 in /opt/anaconda3/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (6.0.4)\n",
      "Requirement already satisfied: yarl<2.0,>=1.0 in /opt/anaconda3/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.9.3)\n",
      "Requirement already satisfied: jinja2 in /opt/anaconda3/lib/python3.12/site-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
      "Requirement already satisfied: jsonschema>=3.0 in /opt/anaconda3/lib/python3.12/site-packages (from altair<6,>=4.0->streamlit) (4.19.2)\n",
      "Requirement already satisfied: toolz in /opt/anaconda3/lib/python3.12/site-packages (from altair<6,>=4.0->streamlit) (0.12.0)\n",
      "Requirement already satisfied: idna>=2.8 in /opt/anaconda3/lib/python3.12/site-packages (from anyio<5,>=3.5.0->openai) (3.7)\n",
      "Collecting marshmallow<4.0.0,>=3.18.0 (from dataclasses-json<0.7,>=0.5.7->langchain_community)\n",
      "  Downloading marshmallow-3.23.1-py3-none-any.whl.metadata (7.5 kB)\n",
      "Collecting typing-inspect<1,>=0.4.0 (from dataclasses-json<0.7,>=0.5.7->langchain_community)\n",
      "  Downloading typing_inspect-0.9.0-py3-none-any.whl.metadata (1.5 kB)\n",
      "Requirement already satisfied: gitdb<5,>=4.0.1 in /opt/anaconda3/lib/python3.12/site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.7)\n",
      "Requirement already satisfied: certifi in /opt/anaconda3/lib/python3.12/site-packages (from httpx<1,>=0.23.0->openai) (2024.7.4)\n",
      "Requirement already satisfied: httpcore==1.* in /opt/anaconda3/lib/python3.12/site-packages (from httpx<1,>=0.23.0->openai) (1.0.2)\n",
      "Requirement already satisfied: h11<0.15,>=0.13 in /opt/anaconda3/lib/python3.12/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai) (0.14.0)\n",
      "Requirement already satisfied: jsonpatch<2.0,>=1.33 in /opt/anaconda3/lib/python3.12/site-packages (from langchain-core<0.4.0,>=0.3.15->langchain) (1.33)\n",
      "Requirement already satisfied: orjson<4.0.0,>=3.9.14 in /opt/anaconda3/lib/python3.12/site-packages (from langsmith<0.2.0,>=0.1.17->langchain) (3.10.11)\n",
      "Requirement already satisfied: requests-toolbelt<2.0.0,>=1.0.0 in /opt/anaconda3/lib/python3.12/site-packages (from langsmith<0.2.0,>=0.1.17->langchain) (1.0.0)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in /opt/anaconda3/lib/python3.12/site-packages (from pandas<3,>=1.3.0->streamlit) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/lib/python3.12/site-packages (from pandas<3,>=1.3.0->streamlit) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/lib/python3.12/site-packages (from pandas<3,>=1.3.0->streamlit) (2023.3)\n",
      "Requirement already satisfied: annotated-types>=0.6.0 in /opt/anaconda3/lib/python3.12/site-packages (from pydantic<3.0.0,>=2.7.4->langchain) (0.6.0)\n",
      "Requirement already satisfied: pydantic-core==2.23.4 in /opt/anaconda3/lib/python3.12/site-packages (from pydantic<3.0.0,>=2.7.4->langchain) (2.23.4)\n",
      "Requirement already satisfied: python-dotenv>=0.21.0 in /opt/anaconda3/lib/python3.12/site-packages (from pydantic-settings<3.0.0,>=2.4.0->langchain_community) (0.21.0)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in /opt/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (2.0.4)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (2.2.2)\n",
      "Requirement already satisfied: markdown-it-py<3.0.0,>=2.2.0 in /opt/anaconda3/lib/python3.12/site-packages (from rich<14,>=10.14.0->streamlit) (2.2.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /opt/anaconda3/lib/python3.12/site-packages (from rich<14,>=10.14.0->streamlit) (2.15.1)\n",
      "Requirement already satisfied: smmap<5,>=3.0.1 in /opt/anaconda3/lib/python3.12/site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in /opt/anaconda3/lib/python3.12/site-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.3)\n",
      "Requirement already satisfied: jsonpointer>=1.9 in /opt/anaconda3/lib/python3.12/site-packages (from jsonpatch<2.0,>=1.33->langchain-core<0.4.0,>=0.3.15->langchain) (2.1)\n",
      "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /opt/anaconda3/lib/python3.12/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.7.1)\n",
      "Requirement already satisfied: referencing>=0.28.4 in /opt/anaconda3/lib/python3.12/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.30.2)\n",
      "Requirement already satisfied: rpds-py>=0.7.1 in /opt/anaconda3/lib/python3.12/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.10.6)\n",
      "Requirement already satisfied: mdurl~=0.1 in /opt/anaconda3/lib/python3.12/site-packages (from markdown-it-py<3.0.0,>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.0)\n",
      "Requirement already satisfied: six>=1.5 in /opt/anaconda3/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas<3,>=1.3.0->streamlit) (1.16.0)\n",
      "Requirement already satisfied: mypy-extensions>=0.3.0 in /opt/anaconda3/lib/python3.12/site-packages (from typing-inspect<1,>=0.4.0->dataclasses-json<0.7,>=0.5.7->langchain_community) (1.0.0)\n",
      "Downloading langchain_community-0.3.7-py3-none-any.whl (2.4 MB)\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m2.4/2.4 MB\u001b[0m \u001b[31m15.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m00:01\u001b[0m00:01\u001b[0m\n",
      "\u001b[?25hDownloading dataclasses_json-0.6.7-py3-none-any.whl (28 kB)\n",
      "Downloading httpx_sse-0.4.0-py3-none-any.whl (7.8 kB)\n",
      "Downloading pydantic_settings-2.6.1-py3-none-any.whl (28 kB)\n",
      "Downloading marshmallow-3.23.1-py3-none-any.whl (49 kB)\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m49.5/49.5 kB\u001b[0m \u001b[31m4.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
      "\u001b[?25hDownloading typing_inspect-0.9.0-py3-none-any.whl (8.8 kB)\n",
      "Installing collected packages: typing-inspect, marshmallow, httpx-sse, dataclasses-json, pydantic-settings, langchain_community\n",
      "Successfully installed dataclasses-json-0.6.7 httpx-sse-0.4.0 langchain_community-0.3.7 marshmallow-3.23.1 pydantic-settings-2.6.1 typing-inspect-0.9.0\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "#importing libraries. Had to import langachain_community separately because it was showing errors.\n",
    "%pip install streamlit langchain langchain_community openai tiktoken"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "e6a4b4b0-4f5b-4261-8d5f-fbc7e6c852df",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "from langchain.llms import OpenAI\n",
    "from langchain.text_splitter import CharacterTextSplitter\n",
    "from langchain.embeddings import OpenAIEmbeddings\n",
    "from langchain.vectorstores import Chroma\n",
    "from langchain.chains import RetrievalQA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "a3aea3bb-9e90-4edd-a8f2-ee916a2d65db",
   "metadata": {},
   "outputs": [],
   "source": [
    "def generate_response(uploaded_file, openai_api_key, query_text):\n",
    "    # Load document if file is uploaded\n",
    "    if uploaded_file is not None:\n",
    "        documents = [uploaded_file.read().decode()]\n",
    "        # Split documents into chunks\n",
    "        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)\n",
    "        texts = text_splitter.create_documents(documents)\n",
    "        # Select embeddings\n",
    "        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)\n",
    "        # Create a vectorstore from documents\n",
    "        db = Chroma.from_documents(texts, embeddings)\n",
    "        # Create retriever interface\n",
    "        retriever = db.as_retriever()\n",
    "        # Create QA chain\n",
    "        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)\n",
    "    return qa.run(query_text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "837acecf-dad8-4d06-a82b-abc1557115d8",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-11-16 14:21:51.552 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /opt/anaconda3/lib/python3.12/site-packages/ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Page title\n",
    "st.set_page_config(page_title='🦜🔗 Ask the Doc App')\n",
    "st.title('🦜🔗 Ask the Doc App')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "a83e7070-cbbe-46a2-b56a-dd473b9a9771",
   "metadata": {},
   "outputs": [],
   "source": [
    "# File upload\n",
    "uploaded_file = st.file_uploader('Upload an article', type='txt')\n",
    "# Query text\n",
    "query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "48191402-e615-4caa-bb2a-bc10ad2da5ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Form input and query\n",
    "result = []\n",
    "with st.form('myform', clear_on_submit=True):\n",
    "    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))\n",
    "    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))\n",
    "    if submitted and openai_api_key.startswith('sk-'):\n",
    "        with st.spinner('Calculating...'):\n",
    "            response = generate_response(uploaded_file, openai_api_key, query_text)\n",
    "            result.append(response)\n",
    "            del openai_api_key\n",
    "\n",
    "if len(result):\n",
    "    st.info(response)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
