from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name : str =  Field(description="Name of the Person")
    age : int = Field(ge=18, descripiton="Age of the Person")
    city : str = Field(description="Name of the city of the Person belong to")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age, and city of a ficional {place} person \n{format_instruction}",
    input_variables = ['place'],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
) 

# prompt = template.format(place="Indian")
# prompt = template.invoke({'place':"Indian"})
# result = model.invoke(prompt)
# result = parser.parse(result.content)

# print(prompt, "\n\n")

chain = template | model | parser
result = chain.invoke({'place':"Indian"})

print(result)
print(type(result))
