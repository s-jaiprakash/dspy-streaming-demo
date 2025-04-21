import dspy
from dspy.streaming import streaming_response
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/stream_example.html")




lm = dspy.LM("gemini/gemini-2.5-flash-preview-04-17")
dspy.configure(lm=lm)
program = dspy.streamify(
    dspy.Predict("context, q -> a")
)  # Signature now includes "context"

# Store streams in a dictionary, keyed by a simple counter (not production-ready)
stream_registry = {}
stream_counter = 0


class QuestionInputWithContext(BaseModel):
    question: str
    context: str = ""  # Optional context


@app.post("/ask")
async def ask_question_post(data: QuestionInputWithContext):
    global stream_counter
    stream_id = f"stream_{stream_counter}"
    stream_counter += 1
    stream_generator = program(
        q=data.question, context=data.context
    )  # Pass context to program
    stream_registry[stream_id] = stream_generator  # Store the stream generator
    return {
        "stream_url": f"/stream/{stream_id}"
    }  # Return URL for client to connect to stream


@app.get("/stream/{stream_id}")
async def stream_output(stream_id: str):
    stream_generator = stream_registry.get(stream_id)
    if stream_generator is None:
        raise HTTPException(status_code=404, detail="Stream not found")

    async def streamer():
        async for chunk in streaming_response(stream_generator):
            yield chunk

    return StreamingResponse(streamer(), media_type="text/event-stream")
