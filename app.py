import gradio as gr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import time
import re


load_dotenv()


# --------------------------------------------
# Core Chain Setup
# --------------------------------------------
def extract_video_id(input_str):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", input_str)
    return match.group(1) if match else input_str

def build_chain_from_video(video_id):
    try:
        # Fetch transcript
        video_id = extract_video_id(video_id)
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_data)

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.create_documents([transcript])

        # Vector store for retrieval
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(docs, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})

        # Prompt + LLM setup
        prompt = PromptTemplate(
            template="""
                You are a helpful assistant.
                ONLY use the context provided from the transcript.
                If the answer is not there, say "I dont know".

                Context:
                {context}

                Question:
                {question}
                """,
            input_variables=["context", "question"]
        )

        def format_docs(docs): return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            })
            | prompt
            | ChatOpenAI(temperature=0.4)
            | StrOutputParser()
        )

        return chain, transcript

    except TranscriptsDisabled:
        raise gr.Error("⚠️ No English transcript available for this video.")
    except Exception as e:
        raise gr.Error(f"❌ Failed to process video: {str(e)}")


# --------------------------------------------
# Query Processing
# --------------------------------------------
def handle_user_query(video_id, user_question):
    try:
        chain, transcript = build_chain_from_video(video_id)
        answer = chain.invoke(user_question)
        return answer.strip(), transcript
    except Exception as err:
        return str(err), ""


# --------------------------------------------
# Gradio UI
# --------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(), css="""
    .gr-box { border-radius: 12px !important; box-shadow: 0 4px 10px rgba(0,0,0,0.07) !important; }
    .title { font-size: 2rem; font-weight: bold; margin-bottom: 0.25em; }
    .subtitle { color: #555; font-size: 1rem; margin-bottom: 1.2em; }
""") as app:
    
    with gr.Column():
        gr.Markdown("<div class='title'>🎥 YouTube ChatBot Assistant</div><div class='subtitle'>Ask questions about any video — powered by OpenAI + LangChain</div>")

        with gr.Row():
            with gr.Column(scale=3):
                video_input = gr.Textbox(label="YouTube Video ID", placeholder="e.g. Gfr50f6ZBvo")
                question_input = gr.Textbox(label="Your Question", placeholder="Ask something about the video...")

                submit = gr.Button("🔍 Get Answer", variant="primary")

            with gr.Column(scale=2):
                answer_box = gr.Textbox(label="🧠 Assistant's Answer", lines=8, interactive=False, show_copy_button=True)
        
        with gr.Accordion("📜 View Full Transcript", open=False):
            transcript_hidden = gr.Textbox(visible=False)
            transcript_box = gr.Textbox(label="Full Transcript", lines=15, interactive=False, show_copy_button=True)

        # Events
        submit.click(
            fn=handle_user_query,
            inputs=[video_input, question_input],
            outputs=[answer_box, transcript_hidden]
        )

        transcript_hidden.change(
            fn=lambda txt: txt,
            inputs=[transcript_hidden],
            outputs=[transcript_box]
        )

if __name__ == "__main__":
    app.launch()
