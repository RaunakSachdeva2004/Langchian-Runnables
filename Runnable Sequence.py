from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()


prompt1 = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="explain the following joke - {text}, in short",
    input_variables=['text']
)

llm = ChatGroq(model="openai/gpt-oss-120b")

parser = StrOutputParser()

chain = RunnableSequence(prompt1, llm, parser, prompt2, llm, parser)



result = chain.invoke({'topic':'AI'})
print(result)
