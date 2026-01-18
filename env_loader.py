import os
from dotenv import load_dotenv
from phoenix.otel import register

load_dotenv()

OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")
PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPEN_AI_KEY is missing in .env file!")

tracer_provider = register(
    project_name="multi-agent-system",
    endpoint=PHOENIX_ENDPOINT,
    auto_instrument=True
)

tracer = tracer_provider.get_tracer(__name__)

print("✅ Environment loaded & Phoenix registered")
