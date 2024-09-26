from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
csv_path = os.path.join('data_csv', 'callahan_time.csv')


def prepromted_csv_agent(csv_file):
    agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-4o"),
        path=csv_path,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True

    )

    result = agent.run('''Give me feedback on the left and right arm angles focusing on these 2 things:
         1) let me know all the times when the imbalance column is more than 10 between the left and right arm 
         2) what is the min and max of both left and right arms?''')
    return print(result)


def csv_chatbot(csv_file):
    # Create a ConversationBufferMemory to store chat history
    memory = ConversationBufferMemory(return_messages=True)

    # Create the CSV agent
    agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-4"),
        path=csv_file,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )

    print("Hey! I'm ready to start answering questions about the CSV file. (Type 'exit' to stop):")
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break

        # Get the chat history
        chat_history = memory.chat_memory.messages

        # Construct the full query with chat history
        full_query = f"Chat History: {chat_history}\nHuman: {query}\nAI Assistant:"

        # Run the agent with the query
        response = agent.run(full_query)

        print(f"AI Assistant: {response}")

        # Add the interaction to the memory
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(response)

csv_chatbot(csv_path)