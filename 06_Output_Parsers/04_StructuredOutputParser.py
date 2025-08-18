from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="Fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="Fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="Fact_3", description="Fact 3 about the topic"),
    ResponseSchema(name="Fact_4", description="Fact 4 about the topic"),
    ResponseSchema(name="Fact_5", description="Fact 5 about the topic"),
    ResponseSchema(name="Fact_6", description="Fact 6 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give the 6 fact about {topic} \n{format_instruction}",
    input_variables = ['topic'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

# prompt = template.format(topic="Changes in the AI research")
# prompt = template.invoke({'topic': "Changes in the AI Research"})
# result = model.invoke(prompt)
# result = parser.parse(result.content)

chain = template | model | parser
result = chain.invoke({'topic': 'Changes in the AI Research'})

print(result)
print(type(result))

# Disadvantage
# - no data validation