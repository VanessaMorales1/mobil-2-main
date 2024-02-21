import {OpenAI } from "langchain";
import {RetrievalQAChain} from "langchain/chains";
import {PDFLoader} from "langchain/document_loaders/fs/pdf";
import {OpenAIEmbeddings} from "langchain/embeddings/openai";
import {MemoryVectorStore} from "langchain/vectorstores/memory";
//import { OpenAI } from "@langchain/openai";




export const process_doc = async (filename: string | undefined, question: string) => {
    const model = new OpenAI({modelName:'gpt-3.5-turbo'
    });//se crea una nueva instancia del modelo de OpenAi

    const loader = new PDFLoader(`/Users/vlmor/Documents/10mo/dispositivos/proyectos/mobil-2-main/backend/uploads/${filename}`, {
        splitPages: false
    }) //se crea una instancia al cargador del pdf,se usa la ruta del archivo
    const doc = await loader.load() //se carga el pdf usando el cargador de pdf, devuelve un objeto que representa el doc pdf cargado
    const vectorStore = await MemoryVectorStore.fromDocuments(doc, new OpenAIEmbeddings()) //se crea un almacen de vectores en memoria a partir de los documentos pdf cargado.conversion de los documentos del pdf a vectores utilizando embeddings de openai
    const vectorStoreRetriever = vectorStore.asRetriever() //se convierte el almacen de vectores en un recuperador de vectore
    const chain = RetrievalQAChain.fromLLM(model, vectorStoreRetriever); //se crea una cadena de recuperacion de preguntas y rescpuestas usando lmn y el recuperador de vectores
    return await chain.call({ //se realiza la llamada a la cadena de recup con la pregunta especificada, devuelve una respuesta a la pregunta basado en el pdf
        query: question,
    })
}
