from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

llm = ChatGroq(model ="openai/gpt-oss-120b")

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text} in 3 lines',
    input_variables=['text']
)

joke_gen_chain = RunnableSequence(prompt1, llm, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, llm, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic':'cricket'}))