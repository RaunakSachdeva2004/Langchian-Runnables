from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough, RunnableBranch

load_dotenv()

llm = ChatGroq(model = "openai/gpt-oss-120b")

prompt1 = PromptTemplate(
    template= "write a detailed report on {topic}",
    input_variables=['topic']
)


prompt2 = PromptTemplate(
    template= "Summarize the following {text}",
    input_variables=['text']
)


parser  = StrOutputParser()


report_gen_chain = RunnableSequence(prompt1, llm, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, prompt2 | llm | parser),
    RunnablePassthrough()
)


final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukraine'}))