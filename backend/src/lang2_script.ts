import {OpenAI} from "langchain";
import {RetrievalQAChain} from "langchain/chains";
import {PDFLoader} from "langchain/document_loaders/fs/pdf";
import {OpenAIEmbeddings} from "langchain/embeddings/openai";
import {MemoryVectorStore} from "langchain/vectorstores/memory";
//import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import RNFS from 'react-native-fs';


import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";


export const new_process = async (filename: string | undefined, question: string) => {
    const model = new OpenAI({modelName:'gpt-3.5-turbo'});
    const documentsPath = RNFS.DocumentDirectoryPath;
   const filePath = `${documentsPath}/${filename}`;
    const loader = new PDFLoader(filePath, {
        splitPages: false
    })
    
    const doc = await loader.load()
    const textSplitter= new RecursiveCharacterTextSplitter({ chunkSize: 200 , chunkOverlap: 100});
    const splitDocs = await textSplitter.splitDocuments(doc)
    const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, new OpenAIEmbeddings())
    const vectorStoreRetriever = vectorStore.asRetriever()
    const chain = RetrievalQAChain.fromLLM(model, vectorStoreRetriever);
   
    return await chain.call({
        query: question,
    })
}
